import unittest

from FWLiteAnalyzer import FWLiteAnalyzer


class TestEvent135509344(unittest.TestCase):
  def setUp(self):
    inputFiles = ["Event135509344_out.root"]
    # 4 2 1 1 1 1 13 6 8 5 0 33
    # 4 2 1 0 3 1 14 9 8 5 1 134
    # 4 2 1 0 4 1 14 9 17 5 1 114

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

    hit = hits[2]
    self.assertEqual(hit.strip      , 33)
    self.assertEqual(hit.wire       , 8)
    self.assertEqual(hit.phi_fp     , 2441)
    self.assertEqual(hit.theta_fp   , 53)
    self.assertEqual((1<<hit.ph_hit), 262144)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[3]
    self.assertEqual(hit.strip      , 134)
    self.assertEqual(hit.wire       , 8)
    self.assertEqual(hit.phi_fp     , 2477)
    self.assertEqual(hit.theta_fp   , 51)
    self.assertEqual((1<<hit.ph_hit), 524288)
    self.assertEqual(hit.phzvl      , 3)

    hit = hits[4]
    self.assertEqual(hit.strip      , 114)
    self.assertEqual(hit.wire       , 17)
    self.assertEqual(hit.phi_fp     , 2397)
    self.assertEqual(hit.theta_fp   , 52)
    self.assertEqual((1<<hit.ph_hit), 262144)
    self.assertEqual(hit.phzvl      , 2)


  def test_tracks(self):
    tracks = self.analyzer.handles["tracks"].product()

    track = tracks[0]
    self.assertEqual(track.rank         , 0x37)
    self.assertEqual(track.ptlut_address, 898932388)
    self.assertEqual(track.gmt_pt       , 11)
    self.assertEqual(track.mode         , 11)
    self.assertEqual(track.gmt_charge   , 1)
    self.assertEqual(track.gmt_quality  , 15)
    self.assertEqual(track.gmt_eta      , 365-512)
    self.assertEqual(track.gmt_phi      , 29)


# ______________________________________________________________________________
if __name__ == "__main__":

  unittest.main()
