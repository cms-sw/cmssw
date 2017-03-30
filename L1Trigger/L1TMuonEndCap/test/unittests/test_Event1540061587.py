import unittest

from FWLiteAnalyzer import FWLiteAnalyzer


class TestEvent1540061587(unittest.TestCase):
  def setUp(self):
    inputFiles = ["Event1540061587_out.root"]
    # 3 1 3 2 1 1 15 10 28 1 0 65
    # 3 1 3 2 1 1 12 5 28 1 1 82
    # 3 1 3 0 2 1 11 2 64 3 0 34

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
    self.assertEqual(hit.wire       , 28)
    self.assertEqual(hit.phi_fp     , 3454)
    self.assertEqual(hit.theta_fp   , 36)
    self.assertEqual((1<<hit.ph_hit), 8192)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[1]
    self.assertEqual(hit.strip      , 82)
    self.assertEqual(hit.wire       , 28)
    self.assertEqual(hit.phi_fp     , 3543)
    self.assertEqual(hit.theta_fp   , 36)
    self.assertEqual((1<<hit.ph_hit), 65536)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[2]
    self.assertEqual(hit.strip      , 34)
    self.assertEqual(hit.wire       , 64)
    self.assertEqual(hit.phi_fp     , 3983)
    self.assertEqual(hit.theta_fp   , 29)
    self.assertEqual((1<<hit.ph_hit), 2048)
    self.assertEqual(hit.phzvl      , 1)

  def test_tracks(self):
    tracks = self.analyzer.handles["tracks"].product()

    # no tracks


# ______________________________________________________________________________
if __name__ == "__main__":

  unittest.main()
