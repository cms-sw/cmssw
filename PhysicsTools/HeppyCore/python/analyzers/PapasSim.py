from PhysicsTools.HeppyCore.framework.analyzer import Analyzer
from PhysicsTools.HeppyCore.papas.simulator import Simulator
from PhysicsTools.HeppyCore.papas.papas_exceptions import PropagationError, SimulationError
from PhysicsTools.HeppyCore.display.core import Display
from PhysicsTools.HeppyCore.display.geometry import GDetector
from PhysicsTools.HeppyCore.display.pfobjects import GTrajectories
from PhysicsTools.HeppyCore.papas.pfalgo.distance  import Distance
from PhysicsTools.HeppyCore.papas.mergedclusterbuilder import MergedClusterBuilder
from PhysicsTools.HeppyCore.papas.data.pfevent import PFEvent
from PhysicsTools.HeppyCore.papas.graphtools.DAG import Node

#todo following Alices merge and reconstruction work
# - add muons and electrons back into the particles, these
#   particles are not yet handled by alices reconstruction
#   they are (for the time being) excluded from the simulation rec particles in order that particle
#   comparisons can be made (eg # no of particles)

class PapasSim(Analyzer):
    '''Runs PAPAS, the PArametrized Particle Simulation.
    
    #This will need to redocumented once new papasdata structure arrives

    Example configuration:

    from PhysicsTools.HeppyCore.analyzers.PapasSim import PapasSim
    from PhysicsTools.HeppyCore.papas.detectors.CMS import CMS
    papas = cfg.Analyzer(
        PapasSim,
        instance_label = 'papas',
        detector = CMS(),
        gen_particles = 'gen_particles_stable',
        sim_particles = 'sim_particles',
        merged_ecals = 'ecal_clusters',
        merged_hcals = 'hcal_clusters',
        tracks = 'tracks',
        #rec_particles = 'sim_rec_particles', # optional - will only do a simulation reconstruction if a name is provided
        output_history = 'history_nodes',
        display_filter_func = lambda ptc: ptc.e()>1.,
        display = False,
        verbose = True
    )
    detector:      Detector model to be used.
    gen_particles: Name of the input gen particle collection
    sim_particles: Name extension for the output sim particle collection.
                   Note that the instance label is prepended to this name.
                   Therefore, in this particular case, the name of the output
                   sim particle collection is "papas_sim_particles".
    merged_ecals: Name for the merged clusters created by simulator
    merged_hcals: Name for the merged clusters created by simulator
    tracks:       Name for smeared tracks created by simulator
    rec_particles: Optional. Name extension for the reconstructed particles created by simulator
                   This is retained for the time being to allow two reconstructions to be compared
                   Reconstruction will occur if this parameter  or rec_particles_no_leptons is provided
                   Same comments as for the sim_particles parameter above.
    rec_particles_no_leptons: Optional. Name extension for the reconstructed particles created by simulator
                   without electrons and muons
                   Reconstruction will occur if this parameter  or rec_particles is provided
                   This is retained for the time being to allow two reconstructions to be compared
                   Same comments as for the sim_particles parameter above.
    smeared: Name for smeared leptons
    history: Optional name for the history nodes, set to None if not needed
    display      : Enable the event display
    verbose      : Enable the detailed printout.

        event must contain
          todo once history is implemented
        event will gain
          ecal_clusters:- smeared merged clusters from simulation
          hcal_clusters:- smeared merged clusters from simulation
          tracks:       - tracks from simulation
          baseline_particles:- simulated particles (excluding electrons and muons)
          sim_particles - simulated particles including electrons and muons
        
    '''

    def __init__(self, *args, **kwargs):
        super(PapasSim, self).__init__(*args, **kwargs)
        self.detector = self.cfg_ana.detector
        self.simulator = Simulator(self.detector, self.mainLogger)
        self.simname = '_'.join([self.instance_label, self.cfg_ana.sim_particles])
        self.tracksname = self.cfg_ana.tracks
        self.mergedecalsname = self.cfg_ana.merged_ecals
        self.mergedhcalsname = self.cfg_ana.merged_hcals
        self.historyname = self.cfg_ana.output_history
        self.is_display = self.cfg_ana.display
        if self.is_display:
            self.init_display()

    def init_display(self):
        self.display = Display(['xy', 'yz'])
        self.gdetector = GDetector(self.detector)
        self.display.register(self.gdetector, layer=0, clearable=False)
        self.is_display = True

    def process(self, event):
        
        event.simulator = self
        if self.is_display:
            self.display.clear()
        pfsim_particles = []
        gen_particles = getattr(event, self.cfg_ana.gen_particles)
        try:
            self.simulator.simulate(gen_particles)
        except (PropagationError, SimulationError) as err:
            self.mainLogger.error(str(err) + ' -> Event discarded')
            return False
        pfsim_particles = self.simulator.ptcs
        if self.is_display  :
            self.display.register(GTrajectories(pfsim_particles),
                                  layer=1)
        #these are the particles before simulation
        simparticles = sorted(pfsim_particles,
                              key=lambda ptc: ptc.e(), reverse=True)
        setattr(event, self.simname, simparticles)

        #extract the tracks and clusters (extraction is prior to Colins merging step)
        event.tracks = dict()
        event.ecal_clusters = dict()
        event.hcal_clusters = dict()
        if "tracker" in self.simulator.pfinput.elements :
            for element in self.simulator.pfinput.elements["tracker"]:
                event.tracks[element.uniqueid] = element

        if "ecal_in" in self.simulator.pfinput.elements :
            for element in self.simulator.pfinput.elements["ecal_in"]:
                event.ecal_clusters[element.uniqueid] = element

        if "hcal_in" in self.simulator.pfinput.elements :
            for element in self.simulator.pfinput.elements["hcal_in"]:
                event.hcal_clusters[element.uniqueid] = element

        ruler = Distance()

        #create history node
        #note eventually history will be created by the simulator and passed in
        # as an argument and this will no longer be needed
        uniqueids = list(event.tracks.keys()) + list(event.ecal_clusters.keys()) + list(event.hcal_clusters.keys())
        history = dict((idt, Node(idt)) for idt in uniqueids)

        #Now merge the simulated clusters and tracks as a separate pre-stage (prior to new reconstruction)
        # and set the event to point to the merged cluster
        pfevent = PFEvent(event, 'tracks', 'ecal_clusters', 'hcal_clusters')
        merged_ecals = MergedClusterBuilder(pfevent.ecal_clusters, ruler, history)
        setattr(event, self.mergedecalsname, merged_ecals.merged)
        merged_hcals = MergedClusterBuilder(pfevent.hcal_clusters, ruler, merged_ecals.history_nodes)
        setattr(event, self.mergedhcalsname, merged_hcals.merged)
        setattr(event, self.historyname, merged_hcals.history_nodes)

        
