import unittest

from FWLiteAnalyzer import FWLiteAnalyzer


class TestEvent1686541662(unittest.TestCase):
  def setUp(self):
    inputFiles = ["Event1686541662_out.root"]
    # 1 2 4 1 1 1 13 7 15 3 1 91
    # 3 2 4 1 1 1 12 4 4 1 0 214
    # 3 2 4 1 1 1 14 8 4 1 0 202
    # 3 2 4 0 2 1 15 10 13 1 0 123
    # 3 2 4 0 3 1 13 7 20 1 1 15
    # 3 2 4 2 5 1 13 6 0 1 0 136

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

    hit = hits[4]
    self.assertEqual(hit.strip      , 91)
    self.assertEqual(hit.wire       , 15)
    self.assertEqual(hit.phi_fp     , 2717)
    self.assertEqual(hit.theta_fp   , 20)
    self.assertEqual((1<<hit.ph_hit), 512)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[5]
    self.assertEqual(hit.strip      , 214-128)
    self.assertEqual(hit.wire       , 4)
    self.assertEqual(hit.phi_fp     , 1403)
    self.assertEqual(hit.theta_fp   , 9)
    self.assertEqual((1<<hit.ph_hit), 32)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[6]
    self.assertEqual(hit.strip      , 202-128)
    self.assertEqual(hit.wire       , 4)
    self.assertEqual(hit.phi_fp     , 1482)
    self.assertEqual(hit.theta_fp   , 9)
    self.assertEqual((1<<hit.ph_hit), 128)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[7]
    self.assertEqual(hit.strip      , 123)
    self.assertEqual(hit.wire       , 13)
    self.assertEqual(hit.phi_fp     , 1604)
    self.assertEqual(hit.theta_fp   , 10)
    self.assertEqual((1<<hit.ph_hit), 4096)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[8]
    self.assertEqual(hit.strip      , 15)
    self.assertEqual(hit.wire       , 20)
    self.assertEqual(hit.phi_fp     , 1446)
    self.assertEqual(hit.theta_fp   , 13)
    self.assertEqual((1<<hit.ph_hit), 128)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[9]
    self.assertEqual(hit.strip      , 136-128)
    self.assertEqual(hit.wire       , 0)
    self.assertEqual(hit.phi_fp     , 1322)
    self.assertEqual(hit.theta_fp   , 7)
    self.assertEqual((1<<hit.ph_hit), 1048576)
    self.assertEqual(hit.phzvl      , 1)

  def test_tracks(self):
    tracks = self.analyzer.handles["tracks"].product()

    track = tracks[0]
    self.assertEqual(track.rank         , 0x3e)
    self.assertEqual(track.ptlut_address, 474472433)
    self.assertEqual(track.gmt_pt       , 5)
    self.assertEqual(track.mode         , 14)
    self.assertEqual(track.gmt_charge   , 1)
    self.assertEqual(track.gmt_quality  , 14)
    self.assertEqual(track.gmt_eta      , 299-512)
    self.assertEqual(track.gmt_phi      , 6)


# ______________________________________________________________________________
if __name__ == "__main__":

  unittest.main()
