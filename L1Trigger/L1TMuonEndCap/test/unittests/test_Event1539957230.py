import unittest

from FWLiteAnalyzer import FWLiteAnalyzer


class TestEvent1539957230(unittest.TestCase):
  def setUp(self):
    inputFiles = ["Event1539957230_out.root"]
    # 3 1 3 2 1 1 13 6 33 2 0 47
    # 3 1 3 0 2 1 15 10 77 2 0 141
    # 3 1 3 0 3 1 12 5 71 3 1 137
    # 3 1 3 0 4 1 13 7 74 3 1 104

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
    self.assertEqual(hit.strip      , 47)
    self.assertEqual(hit.wire       , 33)
    self.assertEqual(hit.phi_fp     , 3964)
    self.assertEqual(hit.theta_fp   , 37)
    self.assertEqual((1<<hit.ph_hit), 1024)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[1]
    self.assertEqual(hit.strip      , 141)
    self.assertEqual(hit.wire       , 77)
    self.assertEqual(hit.phi_fp     , 3644)
    self.assertEqual(hit.theta_fp   , 33)
    self.assertEqual((1<<hit.ph_hit), 274877906944)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[2]
    self.assertEqual(hit.strip      , 137)
    self.assertEqual(hit.wire       , 71)
    self.assertEqual(hit.phi_fp     , 3879)
    self.assertEqual(hit.theta_fp   , 30)
    self.assertEqual((1<<hit.ph_hit), 256)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[3]
    self.assertEqual(hit.strip      , 104)
    self.assertEqual(hit.wire       , 74)
    self.assertEqual(hit.phi_fp     , 4146)
    self.assertEqual(hit.theta_fp   , 27)
    self.assertEqual((1<<hit.ph_hit), 131072)
    self.assertEqual(hit.phzvl      , 1)

  def test_tracks(self):
    tracks = self.analyzer.handles["tracks"].product()

    track = tracks[0]
    self.assertEqual(track.rank         , 0x2a)
    self.assertEqual(track.ptlut_address, 486813690)
    self.assertEqual(track.gmt_pt       , 7)
    self.assertEqual(track.mode         , 14)
    self.assertEqual(track.gmt_charge   , 0)
    self.assertEqual(track.gmt_quality  , 14)
    self.assertEqual(track.gmt_eta      , 169)
    self.assertEqual(track.gmt_phi      , 60)


# ______________________________________________________________________________
if __name__ == "__main__":

  unittest.main()
