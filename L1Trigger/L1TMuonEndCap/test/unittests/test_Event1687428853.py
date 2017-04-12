import unittest

from FWLiteAnalyzer import FWLiteAnalyzer


class TestEvent1687428853(unittest.TestCase):
  def setUp(self):
    inputFiles = ["Event1687428853_out.root"]
    # 3 1 1 0 2 1 14 9 95 1 1 5
    # 3 1 1 0 4 1 15 10 3 4 0 142
    # 3 1 1 2 5 1 13 7 38 1 1 52
    # 3 1 1 2 5 1 15 10 38 1 0 118
    # 3 1 1 0 5 1 15 10 99 4 0 156

    handles = {
      "hits": ("std::vector<L1TMuonEndCap::EMTFHit>", "simEmtfDigisData"),
      "tracks": ("std::vector<L1TMuonEndCap::EMTFTrack>", "simEmtfDigisData"),
    }

    self.analyzer = FWLiteAnalyzer(inputFiles, handles)
    self.analyzer.beginLoop()
    self.event = next(self.analyzer.processLoop())

  def tearDown(self):
    self.analyzer.endLoop()
    self.analyzer = None
    self.event = None

  def test_hits(self):
    hits = self.analyzer.handles["hits"].product()

    hit = hits[0]
    self.assertEqual(hit.strip      , 5)
    self.assertEqual(hit.wire       , 95)
    self.assertEqual(hit.phi_fp     , 1358)
    self.assertEqual(hit.theta_fp   , 42)
    self.assertEqual((1<<hit.ph_hit), 16)
    self.assertEqual(hit.phzvl      , 3)

    hit = hits[1]
    self.assertEqual(hit.strip      , 142)
    self.assertEqual(hit.wire       , 3)
    self.assertEqual(hit.phi_fp     , 1396)
    self.assertEqual(hit.theta_fp   , 43)
    self.assertEqual((1<<hit.ph_hit), 64)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[2]
    self.assertEqual(hit.strip      , 52)
    self.assertEqual(hit.wire       , 38)
    self.assertEqual(hit.phi_fp     , 991)
    self.assertEqual(hit.theta_fp   , 46)
    self.assertEqual((1<<hit.ph_hit), 1024)
    self.assertEqual(hit.phzvl      , 2)

    hit = hits[3]
    self.assertEqual(hit.strip      , 118)
    self.assertEqual(hit.wire       , 38)
    self.assertEqual(hit.phi_fp     , 1325)
    self.assertEqual(hit.theta_fp   , 50)
    self.assertEqual((1<<hit.ph_hit), 1048576)
    self.assertEqual(hit.phzvl      , 2)

    hit = hits[4]
    self.assertEqual(hit.strip      , 156)
    self.assertEqual(hit.wire       , 99)
    self.assertEqual(hit.phi_fp     , 1364)
    self.assertEqual(hit.theta_fp   , 42)
    self.assertEqual((1<<hit.ph_hit), 4398046511104)
    self.assertEqual(hit.phzvl      , 3)

  def test_tracks(self):
    tracks = self.analyzer.handles["tracks"].product()

    #track = tracks[0]
    #self.assertEqual(track.rank         , 0x3d)
    #self.assertEqual(track.ptlut_address, 759184545)
    #self.assertEqual(track.gmt_pt       , 0)
    #self.assertEqual(track.mode         , 13)
    #self.assertEqual(track.gmt_charge   , 1)
    #self.assertEqual(track.gmt_quality  , 13)
    #self.assertEqual(track.gmt_eta      , 157)
    #self.assertEqual(track.gmt_phi      , 0)

    track = tracks[0]
    self.assertEqual(track.rank         , 0x1d)
    self.assertEqual(track.ptlut_address, 693678630)
    self.assertEqual(track.gmt_pt       , 15)
    self.assertEqual(track.mode         , 5)
    self.assertEqual(track.gmt_charge   , 1)
    self.assertEqual(track.gmt_quality  , 5)
    self.assertEqual(track.gmt_eta      , 157)
    self.assertEqual(track.gmt_phi      , 0)


# ______________________________________________________________________________
if __name__ == "__main__":

  unittest.main()
