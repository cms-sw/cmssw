import unittest

from FWLiteAnalyzer import FWLiteAnalyzer


class TestEvent134963582(unittest.TestCase):
  def setUp(self):
    inputFiles = ["Event134963582_out.root"]
    # 4 2 6 2 1 1 13 7 0 1 1 138
    # 4 2 6 0 2 1 15 10 2 2 0 54
    # 4 2 6 0 3 1 14 8 8 2 0 97
    # 4 2 6 0 4 1 15 10 9 2 0 100

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

    hit = hits[7]
    self.assertEqual(hit.strip      , 138-128)
    self.assertEqual(hit.wire       , 0)
    self.assertEqual(hit.phi_fp     , 3706)
    self.assertEqual(hit.theta_fp   , 7)
    self.assertEqual((1<<hit.ph_hit), 2097152)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[8]
    self.assertEqual(hit.strip      , 54)
    self.assertEqual(hit.wire       , 2)
    self.assertEqual(hit.phi_fp     , 3357)
    self.assertEqual(hit.theta_fp   , 7)
    self.assertEqual((1<<hit.ph_hit), 536870912)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[9]
    self.assertEqual(hit.strip      , 97)
    self.assertEqual(hit.wire       , 8)
    self.assertEqual(hit.phi_fp     , 3299)
    self.assertEqual(hit.theta_fp   , 8)
    self.assertEqual((1<<hit.ph_hit), 134217728)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[10]
    self.assertEqual(hit.strip      , 100)
    self.assertEqual(hit.wire       , 9)
    self.assertEqual(hit.phi_fp     , 3325)
    self.assertEqual(hit.theta_fp   , 9)
    self.assertEqual((1<<hit.ph_hit), 268435456)
    self.assertEqual(hit.phzvl      , 1)

  def test_tracks(self):
    tracks = self.analyzer.handles["tracks"].product()

    track = tracks[1]
    self.assertEqual(track.rank         , 0x2d)
    self.assertEqual(track.ptlut_address, 742214779)
    self.assertEqual(track.gmt_pt       , 5)
    self.assertEqual(track.mode         , 13)
    self.assertEqual(track.gmt_charge   , 0)
    self.assertEqual(track.gmt_quality  , 13)
    self.assertEqual(track.gmt_eta      , 292-512)
    self.assertEqual(track.gmt_phi      , 52)


# ______________________________________________________________________________
if __name__ == "__main__":

  unittest.main()
