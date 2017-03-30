import unittest

from FWLiteAnalyzer import FWLiteAnalyzer


class TestEvent135692983(unittest.TestCase):
  def setUp(self):
    inputFiles = ["Event135692983_out.root"]
    # 3 2 6 1 1 1 15 10 2 1 0 172
    # 3 2 6 0 2 1 15 10 7 1 0 114
    # 3 2 6 0 3 1 15 10 5 1 0 45
    # 3 2 6 0 4 1 15 10 3 1 0 45

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

    hit = hits[5]
    self.assertEqual(hit.strip      , 172-128)
    self.assertEqual(hit.wire       , 2)
    self.assertEqual(hit.phi_fp     , 1680)
    self.assertEqual(hit.theta_fp   , 8)
    self.assertEqual((1<<hit.ph_hit), 16384)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[6]
    self.assertEqual(hit.strip      , 114)
    self.assertEqual(hit.wire       , 7)
    self.assertEqual(hit.phi_fp     , 1677)
    self.assertEqual(hit.theta_fp   , 8)
    self.assertEqual((1<<hit.ph_hit), 16384)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[7]
    self.assertEqual(hit.strip      , 45)
    self.assertEqual(hit.wire       , 5)
    self.assertEqual(hit.phi_fp     , 1685)
    self.assertEqual(hit.theta_fp   , 8)
    self.assertEqual((1<<hit.ph_hit), 32768)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[8]
    self.assertEqual(hit.strip      , 45)
    self.assertEqual(hit.wire       , 3)
    self.assertEqual(hit.phi_fp     , 1685)
    self.assertEqual(hit.theta_fp   , 8)
    self.assertEqual((1<<hit.ph_hit), 32768)
    self.assertEqual(hit.phzvl      , 1)

  def test_tracks(self):
    tracks = self.analyzer.handles["tracks"].product()

    track = tracks[1]
    self.assertEqual(track.rank         , 0x6b)
    self.assertEqual(track.ptlut_address, 1010828291)
    self.assertEqual(track.gmt_pt       , 193)
    self.assertEqual(track.mode         , 15)
    self.assertEqual(track.gmt_charge   , 0)
    self.assertEqual(track.gmt_quality  , 15)
    self.assertEqual(track.gmt_eta      , 294-512)
    self.assertEqual(track.gmt_phi      , 8)


# ______________________________________________________________________________
if __name__ == "__main__":

  unittest.main()
