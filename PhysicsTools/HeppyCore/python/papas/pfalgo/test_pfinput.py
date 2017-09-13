import unittest
from ROOT import TVector3 
from PhysicsTools.HeppyCore.papas.pfobjects import Cluster, Particle
from pfinput import merge_clusters

class TestPFInput(unittest.TestCase):

    def test_merge_1(self):
        cluster1 = Cluster(10., TVector3(0., 1., 0.), 0.04,   "ecal_in") #alice added layer (needed for Identifier)
        cluster2 = Cluster(20., TVector3(0., 1., 0.99), 0.06,   "ecal_in")
        merge_clusters([cluster1, cluster2])

if __name__ == '__main__':
    unittest.main()



