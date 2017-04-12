import unittest

from FWLiteAnalyzer import FWLiteAnalyzer


class TestEvent1687288694(unittest.TestCase):
  def setUp(self):
    inputFiles = ["Event1687288694_out.root"]
    # 2 2 5 1 1 1 12 4 15 3 0 65
    # 2 2 5 0 2 1 15 10 29 2 0 87
    # 2 2 5 0 3 1 14 9 25 2 1 64
    # 2 2 5 0 4 1 15 10 26 2 0 55
    # 4 2 5 2 1 1 12 4 10 1 0 150
    # 4 2 5 2 1 1 12 4 8 1 0 150

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

    hit = hits[6]
    self.assertEqual(hit.strip      , 65)
    self.assertEqual(hit.wire       , 15)
    self.assertEqual(hit.phi_fp     , 2852)
    self.assertEqual(hit.theta_fp   , 20)
    self.assertEqual((1<<hit.ph_hit), 8192)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[7]
    self.assertEqual(hit.strip      , 87)
    self.assertEqual(hit.wire       , 29)
    self.assertEqual(hit.phi_fp     , 3091)
    self.assertEqual(hit.theta_fp   , 15)
    self.assertEqual((1<<hit.ph_hit), 2097152)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[8]
    self.assertEqual(hit.strip      , 64)
    self.assertEqual(hit.wire       , 25)
    self.assertEqual(hit.phi_fp     , 3037)
    self.assertEqual(hit.theta_fp   , 14)
    self.assertEqual((1<<hit.ph_hit), 524288)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[9]
    self.assertEqual(hit.strip      , 55)
    self.assertEqual(hit.wire       , 26)
    self.assertEqual(hit.phi_fp     , 2963)
    self.assertEqual(hit.theta_fp   , 15)
    self.assertEqual((1<<hit.ph_hit), 131072)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[10]
    self.assertEqual(hit.strip      , 150-128)
    self.assertEqual(hit.wire       , 10)
    self.assertEqual(hit.phi_fp     , 3632)
    self.assertEqual(hit.theta_fp   , 20)
    self.assertEqual((1<<hit.ph_hit), 524288)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[11]
    self.assertEqual(hit.strip      , 150-128)
    self.assertEqual(hit.wire       , 8)
    self.assertEqual(hit.phi_fp     , 3632)
    self.assertEqual(hit.theta_fp   , 18)
    self.assertEqual((1<<hit.ph_hit), 524288)
    self.assertEqual(hit.phzvl      , 1)

  def test_tracks(self):
    tracks = self.analyzer.handles["tracks"].product()

    track = tracks[1]
    self.assertEqual(track.rank         , 0x2a)
    self.assertEqual(track.ptlut_address, 477498879)
    self.assertEqual(track.gmt_pt       , 5)
    self.assertEqual(track.mode         , 14)
    self.assertEqual(track.gmt_charge   , 0)
    self.assertEqual(track.gmt_quality  , 14)
    self.assertEqual(track.gmt_eta      , 310-512)
    self.assertEqual(track.gmt_phi      , 45)


# ______________________________________________________________________________
if __name__ == "__main__":

  unittest.main()
