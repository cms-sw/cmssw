import unittest
from pfobjects import Cluster, SmearedCluster
from detectors.CMS import cms
from simulator import Simulator
from ROOT import TVector3
import math
import numpy as np
from ROOT import TFile, TH1F, TH2F

simulator = Simulator(cms)

class TestCluster(unittest.TestCase):

    def test_pt(self):
        '''Test that pT is correctly set.'''
        cluster = Cluster(10., TVector3(1,0,0), 1)  #alice made this use default layer
        self.assertAlmostEqual(cluster.pt, 10.)
        cluster.set_energy(5.)
        self.assertAlmostEqual(cluster.pt, 5.)

    def test_smear(self):
        rootfile = TFile('test_cluster_smear.root', 'recreate')
        h_e = TH1F('h_e','cluster energy', 200, 5, 15.)
        energy = 10.
        cluster = Cluster(energy, TVector3(1,0,0), 1) #alice made this use default layer
        ecal = cms.elements['ecal']
        energies = []
        for i in range(10000):
            smeared = simulator.smear_cluster(cluster, ecal, accept=True)
            h_e.Fill(smeared.energy)
            energies.append(smeared.energy)
        npe = np.array(energies)
        mean = np.mean(npe)
        rms = np.std(npe)
        eres = ecal.energy_resolution(cluster.energy)
        self.assertAlmostEqual(mean, energy, places=1)
        self.assertAlmostEqual(rms, eres*energy, places=1)
        rootfile.Write()
        rootfile.Close()
        
    def test_acceptance(self):
        rootfile = TFile('test_cluster_acceptance.root', 'recreate')
        h_evseta = TH2F('h_evseta','cluster energy vs eta',
                        100, -5, 5, 100, 0, 15)
        h_ptvseta = TH2F('h_ptvseta','cluster pt vs eta',
                         100, -5, 5, 100, 0, 15)
        nclust = 1000.
        # making 1000 deposits between 0 and 10 GeV
        energies = np.random.uniform(0., 10., nclust)
        # theta between 0 and pi
        thetas = np.random.uniform(0, math.pi, nclust)
        costhetas = np.cos(thetas)
        sinthetas = np.sin(thetas)
        clusters = []
        for energy, cos, sin in zip(energies, costhetas, sinthetas):
            clusters.append(Cluster(energy, TVector3(sin,0,cos), 1))  #alice made this use default layer
        ecal = cms.elements['ecal']
        smeared_clusters = []
        min_energy = -999.
        for cluster in clusters:
            smeared_cluster = simulator.smear_cluster(cluster, ecal)
            if smeared_cluster:
                h_evseta.Fill(smeared_cluster.position.Eta(),
                              smeared_cluster.energy)
                h_ptvseta.Fill(smeared_cluster.position.Eta(),
                               smeared_cluster.pt)
                smeared_clusters.append(smeared_cluster)
                if smeared_cluster.energy > min_energy:
                    min_energy = smeared_cluster.energy
        # test that some clusters have been rejected
        # (not passing the acceptance)
        self.assertGreater(len(clusters), len(smeared_clusters))
        # test that the minimum cluster energy is larger than the
        # minimum ecal threshold
        ecal_min_thresh = min(ecal.emin.values())
        self.assertGreater(min_energy, ecal_min_thresh)
        rootfile.Write()
        rootfile.Close()

    # def test_absorption(self):
    #     energies = [10, 20, 30, 40]
    #     e1, e2, e3, e4 = energies
    #     dists = [-0.01, 0.089, 0.11, 0.16]
    #     sizes  = [0.04, 0.06, 0.06, 0.06]
    #     def make_clusters(proj='z'):
    #         clusters = []
    #         for i, energy in enumerate(energies):
    #             # moving along z, at phi=0.
    #             x, y, z = 1, 0, dists[i]
    #             if proj=='x':
    #                 #moving along x, around phi = pi/2.
    #                 x, y, z = dists[i], 1, 0.
    #             elif proj=='y':
    #                 #moving along y, around phi=0. testing periodic condition. 
    #                 x, y, z = 1, dists[i], 0
    #             position = TVector3(x, y, z)
    #             clusters.append( Cluster( energy,
    #                                       position,
    #                                       sizes[i],
    #                                       0 ))
    #         return clusters
    #     def test(proj):
    #         # test simple absorption between two single clusters
    #         c1, c2, c3, c4 = make_clusters(proj)
    #         print c1.position.X(), c1.position.Y(), c1.position.Z()
    #         print c2.position.X(), c2.position.Y(), c2.position.Z()            
    #         c1.absorb(c2)
    #         self.assertEqual(len(c1.absorbed), 1)
    #         self.assertEqual(len(c2.absorbed), 0)
    #         self.assertEqual(c1.absorbed[0], c2)
    #         self.assertEqual(c1.energy, e1+e2)
    #         # testing absorption of an additional cluster by a compound cluster
    #         c1.absorb(c3)
    #         self.assertEqual(len(c1.absorbed), 2)
    #         self.assertEqual(c1.energy, e1+e2+e3)
    #         c1, c2, c3, c4 = make_clusters(proj)
    #         # testing impossible absorption, cause the 2 clusters are too far
    #         code = c1.absorb(c3)
    #         self.assertFalse(code)
    #         self.assertEqual(len(c1.absorbed), 0)
    #         self.assertEqual(len(c2.absorbed), 0)
    #         self.assertEqual(c1.energy, e1)
    #         c1, c2, c3, c4 = make_clusters(proj)
    #         # testing absorption between two compound clusters
    #         c1.absorb(c2)
    #         self.assertEqual(c1.energy, e1+e2)        
    #         c3.absorb(c4)
    #         self.assertEqual(c3.energy, e3+e4)        
    #         c1.absorb(c3)
    #         self.assertEqual(len(c1.absorbed), 3)
    #         self.assertEqual(c1.energy, e1+e2+e3+e4)
    #     test('z')
    #     test('y')
    #     test('x')
        
if __name__ == '__main__':
    unittest.main()

