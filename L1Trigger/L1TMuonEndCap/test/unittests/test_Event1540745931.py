import unittest

from FWLiteAnalyzer import FWLiteAnalyzer


class TestEvent1540745931(unittest.TestCase):
  def setUp(self):
    inputFiles = ["Event1540745931_out.root"]
    # 3 1 1 2 1 1 15 10 53 5 0 124
    # 3 1 1 0 2 1 15 10 24 8 0 130
    # 3 1 1 0 2 1 14 8 24 8 0 119
    # 3 1 1 0 3 1 15 10 36 8 0 26
    # 3 1 1 0 4 1 14 9 46 8 1 25

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
    self.assertEqual(hit.strip      , 124)
    self.assertEqual(hit.wire       , 53)
    self.assertEqual(hit.phi_fp     , 4228)
    self.assertEqual(hit.theta_fp   , 77)
    self.assertEqual((1<<hit.ph_hit), 262144)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[1]
    self.assertEqual(hit.strip      , 130)
    self.assertEqual(hit.wire       , 24)
    self.assertEqual(hit.phi_fp     , 4252)
    self.assertEqual(hit.theta_fp   , 76)
    self.assertEqual((1<<hit.ph_hit), 524288)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[2]
    self.assertEqual(hit.strip      , 119)
    self.assertEqual(hit.wire       , 24)
    self.assertEqual(hit.phi_fp     , 4207)
    self.assertEqual(hit.theta_fp   , 76)
    self.assertEqual((1<<hit.ph_hit), 131072)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[3]
    self.assertEqual(hit.strip      , 26)
    self.assertEqual(hit.wire       , 36)
    self.assertEqual(hit.phi_fp     , 4260)
    self.assertEqual(hit.theta_fp   , 78)
    self.assertEqual((1<<hit.ph_hit), 524288)
    self.assertEqual(hit.phzvl      , 2)

    hit = hits[4]
    self.assertEqual(hit.strip      , 25)
    self.assertEqual(hit.wire       , 46)
    self.assertEqual(hit.phi_fp     , 4263)
    self.assertEqual(hit.theta_fp   , 78)
    self.assertEqual((1<<hit.ph_hit), 1048576)
    self.assertEqual(hit.phzvl      , 2)

  def test_tracks(self):
    tracks = self.analyzer.handles["tracks"].product()

    track = tracks[0]
    self.assertEqual(track.rank         , 0x6b)
    self.assertEqual(track.ptlut_address, 1047278616)
    self.assertEqual(track.gmt_pt       , 46)
    self.assertEqual(track.mode         , 15)
    self.assertEqual(track.gmt_charge   , 1)
    self.assertEqual(track.gmt_quality  , 15)
    self.assertEqual(track.gmt_eta      , 120)
    self.assertEqual(track.gmt_phi      , 76)


# ______________________________________________________________________________
if __name__ == "__main__":

  unittest.main()
