import unittest
import copy
from merger import merge_clusters
from PhysicsTools.HeppyCore.papas.pfobjects import Cluster
from ROOT import TVector3

class TestMerger(unittest.TestCase):

    def test_merge_pair(self):
        clusters = [ Cluster(20, TVector3(1, 0, 0), 0.1, 'hcal_in'),
                     Cluster(20, TVector3(1.,0.05,0.), 0.1, 'hcal_in')]
        merged_clusters = merge_clusters(clusters, 'hcal_in')
        self.assertEqual( len(merged_clusters), 1 )
        self.assertEqual( merged_clusters[0].energy,
                          clusters[0].energy + clusters[1].energy)
        self.assertEqual( merged_clusters[0].position.X(),
                          (clusters[0].position.X() + clusters[1].position.X())/2.)
        self.assertEqual( len(merged_clusters[0].subclusters), 2)
        self.assertEqual( merged_clusters[0].subclusters[0], clusters[0])
        self.assertEqual( merged_clusters[0].subclusters[1], clusters[1])

        
    def test_merge_pair_away(self):
        clusters = [ Cluster(20, TVector3(1,0,0), 0.04, 'hcal_in'),
                     Cluster(20, TVector3(1,1.1,0.0), 0.04, 'hcal_in')]
        merge_clusters(clusters, 'hcal_in')
        self.assertEqual( len(clusters), 2 )
        self.assertEqual( len(clusters[0].subclusters), 1)
        self.assertEqual( len(clusters[1].subclusters), 1)

    def test_merge_different_layers(self):
        clusters = [ Cluster(20, TVector3(1,0,0), 0.04, 'ecal_in'),
                     Cluster(20, TVector3(1,0,0), 0.04, 'hcal_in')]
        merge_clusters(clusters, 'hcal_in')
        self.assertEqual( len(clusters), 2)

    def test_inside(self):
        clusters = [ Cluster(20, TVector3(1, 0, 0), 0.055, 'hcal_in'),
                     Cluster(20, TVector3(1.,0.1, 0.0), 0.055, 'hcal_in')]
        merged_clusters = merge_clusters(clusters, 'hcal_in')
        self.assertEqual( len(merged_clusters), 1 )
        cluster = merged_clusters[0]
        self.assertEqual( (True, 0.), cluster.is_inside(TVector3(1, 0 , 0)) )
        self.assertEqual( (True, 0.), cluster.is_inside(TVector3(1, 0.1, 0)) )
        in_the_middle = cluster.is_inside(TVector3(1, 0.06, 0))
        self.assertTrue(in_the_middle[0])
        self.assertAlmostEqual(in_the_middle[1], 0.04000)
        self.assertFalse( cluster.is_inside(TVector3(1, 0.156, 0))[0]  )
        

if __name__ == '__main__':
    unittest.main()
