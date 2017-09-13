import sys
import copy
import shelve
from PhysicsTools.HeppyCore.papas.propagator import StraightLinePropagator, HelixPropagator
from PhysicsTools.HeppyCore.papas.pfobjects import Cluster, SmearedCluster, SmearedTrack
from PhysicsTools.HeppyCore.papas.pfobjects import Particle as PFSimParticle
from PhysicsTools.HeppyCore.papas.pfalgo.pfinput import  PFInput

import PhysicsTools.HeppyCore.papas.multiple_scattering as mscat
from PhysicsTools.HeppyCore.papas.papas_exceptions import SimulationError
from PhysicsTools.HeppyCore.utils.pdebug import pdebugger
import PhysicsTools.HeppyCore.statistics.rrandom as random



def pfsimparticle(ptc):
    '''Create a PFSimParticle from a particle.
    The PFSimParticle will have the same p4, vertex, charge, pdg ID.
    '''
    tp4 = ptc.p4()
    vertex = ptc.start_vertex().position()
    charge = ptc.q()
    pid = ptc.pdgid()
    simptc = PFSimParticle(tp4, vertex, charge, pid)
    pdebugger.info(" ".join(("Made", simptc.__str__())))
    simptc.gen_ptc = ptc
    return simptc

class Simulator(object):

    def __init__(self, detector, logger=None):
        self.verbose = True
        self.detector = detector
        if logger is None:
            import logging
            logging.basicConfig(level='ERROR')
            logger = logging.getLogger('Simulator')
        self.logger = logger
        self.prop_helix = HelixPropagator()
        self.prop_straight = StraightLinePropagator()

    def write_ptcs(self, dbname):
        db = shelve.open(dbname)
        db['ptcs'] = self.ptcs
        db.close()

    def reset(self):
        self.particles = None
        self.ptcs = None
        Cluster.max_energy = 0.
        SmearedCluster.max_energy = 0.

    def propagator(self, ptc):
        is_neutral = abs(ptc.q()) < 0.5
        return self.prop_straight if is_neutral else self.prop_helix

    def propagate(self, ptc):
        '''propagate the particle to all detector cylinders'''
        self.propagator(ptc).propagate([ptc], self.detector.cylinders(),
                                       self.detector.elements['field'].magnitude)

    def make_cluster(self, ptc, detname, fraction=1., size=None):
        '''adds a cluster in a given detector, with a given fraction of
        the particle energy.'''
        detector = self.detector.elements[detname]
        self.propagator(ptc).propagate_one(ptc,
                                           detector.volume.inner,
                                           self.detector.elements['field'].magnitude)
        if size is None:
            size = detector.cluster_size(ptc)
        cylname = detector.volume.inner.name
        if not cylname in ptc.points:
            # TODO Colin particle was not extrapolated here...
            # issue must be solved!
            errormsg = '''
SimulationError : cannot make cluster for particle: 
particle: {ptc}
with vertex rho={rho:5.2f}, z={zed:5.2f}
cannot be extrapolated to : {det}\n'''.format(ptc=ptc,
                                              rho=ptc.vertex.Perp(),
                                              zed=ptc.vertex.Z(),
                                              det=detector.volume.inner)
            self.logger.warning(errormsg)
            raise SimulationError('Particle not extrapolated to the detector, so cannot make a cluster there. No worries for now, problem will be solved :-)')
        cluster = Cluster(ptc.p4().E()*fraction, ptc.points[cylname], size, cylname, ptc)
        ptc.clusters[cylname] = cluster
        pdebugger.info(" ".join(("Made", cluster.__str__())))
        return cluster

    def smear_cluster(self, cluster, detector, accept=False, acceptance=None):
        '''Returns a copy of self with a smeared energy.
        If accept is False (default), returns None if the smeared
        cluster is not in the detector acceptance. '''

        eres = detector.energy_resolution(cluster.energy, cluster.position.Eta())
        response = detector.energy_response(cluster.energy, cluster.position.Eta())
        energy = cluster.energy * random.gauss(response, eres)
        smeared_cluster = SmearedCluster(cluster,
                                         energy,
                                         cluster.position,
                                         cluster.size(),
                                         cluster.layer,
                                         cluster.particle)
        pdebugger.info(str('Made {}'.format(smeared_cluster)))
        det = acceptance if acceptance else detector
        if det.acceptance(smeared_cluster) or accept:
            return smeared_cluster
        else:
            pdebugger.info(str('Rejected {}'.format(smeared_cluster)))
            return None

    def smear_track(self, track, detector, accept=False):
        #TODO smearing depends on particle type!
        ptres = detector.pt_resolution(track)
        scale_factor = random.gauss(1, ptres)
        smeared_track = SmearedTrack(track,
                                     track.p3 * scale_factor,
                                     track.charge,
                                     track.path)
        pdebugger.info(" ".join(("Made", smeared_track.__str__())))
        if detector.acceptance(smeared_track) or accept:
            return smeared_track
        else:
            pdebugger.info(str('Rejected {}'.format(smeared_track)))
            return None

    def simulate_photon(self, ptc):
        pdebugger.info("Simulating Photon")
        detname = 'ecal'
        ecal = self.detector.elements[detname]
        self.prop_straight.propagate_one(ptc,
                                         ecal.volume.inner)

        cluster = self.make_cluster(ptc, detname)
        smeared = self.smear_cluster(cluster, ecal)
        if smeared:
            ptc.clusters_smeared[smeared.layer] = smeared


    def simulate_electron(self, ptc):
        pdebugger.info("Simulating Electron")
        ecal = self.detector.elements['ecal']
        self.prop_helix.propagate_one(ptc,
                                      ecal.volume.inner,
                                      self.detector.elements['field'].magnitude)
        cluster = self.make_cluster(ptc, 'ecal')
        smeared_cluster = self.smear_cluster(cluster, ecal)
        if smeared_cluster:
            ptc.clusters_smeared[smeared_cluster.layer] = smeared_cluster
        smeared_track = self.smear_track(ptc.track,
                                         self.detector.elements['tracker'])
        if smeared_track:
            ptc.track_smeared = smeared_track


    def simulate_neutrino(self, ptc):
        self.propagate(ptc)

    def simulate_hadron(self, ptc):
        '''Simulate a hadron, neutral or charged.
        ptc should behave as pfobjects.Particle.
        '''
        pdebugger.info("Simulating Hadron")
        #implement beam pipe scattering

        ecal = self.detector.elements['ecal']
        hcal = self.detector.elements['hcal']
        beampipe = self.detector.elements['beampipe']
        frac_ecal = 0.

        self.propagator(ptc).propagate_one(ptc,
                                           beampipe.volume.inner,
                                           self.detector.elements['field'].magnitude)

        self.propagator(ptc).propagate_one(ptc,
                                           beampipe.volume.outer,
                                           self.detector.elements['field'].magnitude)

        mscat.multiple_scattering(ptc, beampipe, self.detector.elements['field'].magnitude)

        #re-propagate after multiple scattering in the beam pipe
        #indeed, multiple scattering is applied within the beam pipe,
        #so the extrapolation points to the beam pipe entrance and exit
        #change after multiple scattering.
        self.propagator(ptc).propagate_one(ptc,
                                           beampipe.volume.inner,
                                           self.detector.elements['field'].magnitude)
        self.propagator(ptc).propagate_one(ptc,
                                           beampipe.volume.outer,
                                           self.detector.elements['field'].magnitude)
        self.propagator(ptc).propagate_one(ptc,
                                           ecal.volume.inner,
                                           self.detector.elements['field'].magnitude)

        # these lines moved earlier in order to match cpp logic
        if ptc.q() != 0:
            pdebugger.info(" ".join(("Made", ptc.track.__str__())))
            smeared_track = self.smear_track(ptc.track,
                                             self.detector.elements['tracker'])
            if smeared_track:
                ptc.track_smeared = smeared_track

        if 'ecal_in' in ptc.path.points:
            # doesn't have to be the case (long-lived particles)
            path_length = ecal.material.path_length(ptc)
            if path_length < sys.float_info.max:
                # ecal path length can be infinite in case the ecal
                # has lambda_I = 0 (fully transparent to hadrons)
                time_ecal_inner = ptc.path.time_at_z(ptc.points['ecal_in'].Z())
                deltat = ptc.path.deltat(path_length)
                time_decay = time_ecal_inner + deltat
                point_decay = ptc.path.point_at_time(time_decay)
                ptc.points['ecal_decay'] = point_decay
                if ecal.volume.contains(point_decay):
                    frac_ecal = random.uniform(0., 0.7)
                    cluster = self.make_cluster(ptc, 'ecal', frac_ecal)
                    # For now, using the hcal resolution and acceptance
                    # for hadronic cluster
                    # in the ECAL. That's not a bug!
                    smeared = self.smear_cluster(cluster, hcal, acceptance=ecal)
                    if smeared:
                        ptc.clusters_smeared[smeared.layer] = smeared

        cluster = self.make_cluster(ptc, 'hcal', 1-frac_ecal)
        smeared = self.smear_cluster(cluster, hcal)
        if smeared:
            ptc.clusters_smeared[smeared.layer] = smeared

    def simulate_muon(self, ptc):
        pdebugger.info("Simulating Muon")
        self.propagate(ptc)
        smeared_track = self.smear_track(ptc.track,
                                         self.detector.elements['tracker'])
        if smeared_track:
            ptc.track_smeared = smeared_track

    def smear_muon(self, ptc):
        pdebugger.info("Smearing Muon")
        self.propagate(ptc)
        if ptc.q() != 0:
            pdebugger.info(" ".join(("Made", ptc.track.__str__())))
        smeared = copy.deepcopy(ptc)
        return smeared

    def smear_electron(self, ptc):
        pdebugger.info("Smearing Electron")
        ecal = self.detector.elements['ecal']
        self.prop_helix.propagate_one(ptc,
                                      ecal.volume.inner,
                                      self.detector.elements['field'].magnitude)
        if ptc.q() != 0:
            pdebugger.info(" ".join(("Made", ptc.track.__str__())))
        smeared = copy.deepcopy(ptc)
        return smeared

    def propagate_muon(self, ptc):
        pdebugger.info("Propogate Muon")
        self.propagate(ptc)
        return

    def propagate_electron(self, ptc):
        pdebugger.info("Propogate Electron")
        ecal = self.detector.elements['ecal']
        self.prop_helix.propagate_one(ptc,
                                      ecal.volume.inner,
                                      self.detector.elements['field'].magnitude)
        return

    def simulate(self, ptcs):
        self.reset()
        self.ptcs = []

        #newsort
        for gen_ptc in sorted(ptcs, key=lambda ptc: ptc.uniqueid):
            pdebugger.info(str('{}'.format(gen_ptc)))
        for gen_ptc in ptcs:
            ptc = pfsimparticle(gen_ptc)
            if ptc.pdgid() == 22:
                self.simulate_photon(ptc)
            elif abs(ptc.pdgid()) == 11: #check with colin
                self.propagate_electron(ptc)
                #smeared_ptc = self.smear_electron(ptc)
                #smeared.append(smeared_ptc)
                # self.simulate_electron(ptc)
            elif abs(ptc.pdgid()) == 13:   #check with colin
                self.propagate_muon(ptc)
                #smeared_ptc = self.smear_muon(ptc)
                #smeared.append(smeared_ptc)
                # self.simulate_muon(ptc)
            elif abs(ptc.pdgid()) in [12, 14, 16]:
                self.simulate_neutrino(ptc)
            elif abs(ptc.pdgid()) > 100: #TODO make sure this is ok
                if ptc.q() and ptc.pt() < 0.2:
                    # to avoid numerical problems in propagation
                    continue
                self.simulate_hadron(ptc)
            self.ptcs.append(ptc)
            self.pfinput = PFInput(self.ptcs) #collect up tracks, clusters etc ready for merging/reconstruction_muon(otc)

if __name__ == '__main__':

    import math
    import logging
    from detectors.CMS import cms
    from toyevents import particle
    from PhysicsTools.HeppyCore.display.core import Display
    from PhysicsTools.HeppyCore.display.geometry import GDetector
    from PhysicsTools.HeppyCore.display.pfobjects import GTrajectories

    display_on = True
    detector = cms

    logging.basicConfig(level='WARNING')
    logger = logging.getLogger('Simulator')
    logger.addHandler(logging.StreamHandler(sys.stdout))

    for i in range(1):
        if not i%100:
            print i
        simulator = Simulator(detector, logger)
        # particles = monojet([211, -211, 130, 22, 22, 22], math.pi/2., math.pi/2., 2, 50)
        particles = [
            # particle(211, math.pi/2., math.pi/2., 100),
            particle(211, math.pi/2 + 0.5, 0., 40.),
            # particle(130, math.pi/2., math.pi/2.+0., 100.),
            # particle(22, math.pi/2., math.pi/2.+0.0, 10.)
        ]
        simulator.simulate(particles)

    if display_on:
        display = Display(['xy', 'yz',
                           'ECAL_thetaphi',
                           'HCAL_thetaphi'
                           ])
        gdetector = GDetector(detector)
        display.register(gdetector, 0)
        gtrajectories = GTrajectories(simulator.ptcs)
        display.register(gtrajectories, 1)
        display.draw()

