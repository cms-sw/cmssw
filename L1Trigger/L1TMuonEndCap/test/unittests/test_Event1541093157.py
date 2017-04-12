import unittest

from FWLiteAnalyzer import FWLiteAnalyzer


class TestEvent1541093157(unittest.TestCase):
  def setUp(self):
    inputFiles = ["Event1541093157_out.root"]
    # 3 1 2 1 1 1 12 5 11 5 1 65
    # 3 1 2 0 3 1 14 8 5 5 0 47
    # 3 1 2 0 4 1 14 8 10 5 0 65
    # 7 1 2 1 1 1 14 8 0 2 0 196

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
    self.assertEqual(hit.strip      , 65)
    self.assertEqual(hit.wire       , 11)
    self.assertEqual(hit.phi_fp     , 2194)
    self.assertEqual(hit.theta_fp   , 55)
    self.assertEqual((1<<hit.ph_hit), 2048)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[1]
    self.assertEqual(hit.strip      , 47)
    self.assertEqual(hit.wire       , 5)
    self.assertEqual(hit.phi_fp     , 2377)
    self.assertEqual(hit.theta_fp   , 49)
    self.assertEqual((1<<hit.ph_hit), 65536)
    self.assertEqual(hit.phzvl      , 3)

    hit = hits[2]
    self.assertEqual(hit.strip      , 65)
    self.assertEqual(hit.wire       , 10)
    self.assertEqual(hit.phi_fp     , 2305)
    self.assertEqual(hit.theta_fp   , 47)
    self.assertEqual((1<<hit.ph_hit), 32768)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[3]
    self.assertEqual(hit.strip      , 196-128)
    self.assertEqual(hit.wire       , 0)
    self.assertEqual(hit.phi_fp     , 2382)
    self.assertEqual(hit.theta_fp   , 7)
    self.assertEqual((1<<hit.ph_hit), 131072)
    self.assertEqual(hit.phzvl      , 1)

  def test_tracks(self):
    tracks = self.analyzer.handles["tracks"].product()

    track = tracks[0]
    self.assertEqual(track.rank         , 0x13)
    self.assertEqual(track.ptlut_address, 830547016)
    self.assertEqual(track.gmt_pt       , 9)
    self.assertEqual(track.mode         , 3)
    self.assertEqual(track.gmt_charge   , 0)
    self.assertEqual(track.gmt_quality  , 7)
    self.assertEqual(track.gmt_eta      , 148)
    self.assertEqual(track.gmt_phi      , 27)


# ______________________________________________________________________________
if __name__ == "__main__":

  unittest.main()
