import unittest

from FWLiteAnalyzer import FWLiteAnalyzer


class TestEvent1686648178(unittest.TestCase):
  def setUp(self):
    inputFiles = ["Event1686648178_out.root"]
    # 3 1 4 2 1 1 15 10 27 3 0 86
    # 3 1 4 0 2 1 15 10 84 3 0 134
    # 3 1 4 0 3 1 15 10 84 3 0 24
    # 3 1 4 2 5 1 14 8 33 1 0 109
    # 3 1 4 0 5 1 15 10 96 4 0 114
    # 3 1 4 0 5 1 15 10 6 9 0 71

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

    hit = hits[9]
    self.assertEqual(hit.strip      , 86)
    self.assertEqual(hit.wire       , 27)
    self.assertEqual(hit.phi_fp     , 4764)
    self.assertEqual(hit.theta_fp   , 36)
    self.assertEqual((1<<hit.ph_hit), 131072)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[10]
    self.assertEqual(hit.strip      , 134)
    self.assertEqual(hit.wire       , 84)
    self.assertEqual(hit.phi_fp     , 4788)
    self.assertEqual(hit.theta_fp   , 36)
    self.assertEqual((1<<hit.ph_hit), 137438953472)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[11]
    self.assertEqual(hit.strip      , 24)
    self.assertEqual(hit.wire       , 84)
    self.assertEqual(hit.phi_fp     , 4788)
    self.assertEqual(hit.theta_fp   , 36)
    self.assertEqual((1<<hit.ph_hit), 137438953472)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[12]
    self.assertEqual(hit.strip      , 109)
    self.assertEqual(hit.wire       , 33)
    self.assertEqual(hit.phi_fp     , 1277)
    self.assertEqual(hit.theta_fp   , 44)
    self.assertEqual((1<<hit.ph_hit), 524288)
    self.assertEqual(hit.phzvl      , 2)

    hit = hits[13]
    self.assertEqual(hit.strip      , 114)
    self.assertEqual(hit.wire       , 96)
    self.assertEqual(hit.phi_fp     , 1028)
    self.assertEqual(hit.theta_fp   , 43)
    self.assertEqual((1<<hit.ph_hit), 2147483648)
    self.assertEqual(hit.phzvl      , 3)

    hit = hits[14]
    self.assertEqual(hit.strip      , 71)
    self.assertEqual(hit.wire       , 6)
    self.assertEqual(hit.phi_fp     , 1080)
    self.assertEqual(hit.theta_fp   , 43)
    self.assertEqual((1<<hit.ph_hit), 16384)
    self.assertEqual(hit.phzvl      , 1)

  def test_tracks(self):
    tracks = self.analyzer.handles["tracks"].product()

    track = tracks[2]
    self.assertEqual(track.rank         , 0x6a)
    self.assertEqual(track.ptlut_address, 489746456)
    self.assertEqual(track.gmt_pt       , 116)
    self.assertEqual(track.mode         , 14)
    self.assertEqual(track.gmt_charge   , 1)
    self.assertEqual(track.gmt_quality  , 14)
    self.assertEqual(track.gmt_eta      , 165)
    self.assertEqual(track.gmt_phi      , 90)

    track = tracks[3]
    self.assertEqual(track.rank         , 0x39)
    self.assertEqual(track.ptlut_address, 759344630)
    self.assertEqual(track.gmt_pt       , 11)
    self.assertEqual(track.mode         , 13)
    self.assertEqual(track.gmt_charge   , 0)
    self.assertEqual(track.gmt_quality  , 13)
    self.assertEqual(track.gmt_eta      , 156)
    self.assertEqual(track.gmt_phi      , 247-256)


# ______________________________________________________________________________
if __name__ == "__main__":

  unittest.main()
