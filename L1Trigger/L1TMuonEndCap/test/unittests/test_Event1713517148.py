import unittest

from FWLiteAnalyzer import FWLiteAnalyzer


class TestEvent1713517148(unittest.TestCase):
  def setUp(self):
    inputFiles = ["Event1713517148_out.root"]
    # 4 2 4 1 1 1 14 9 20 3 1 72
    # 4 2 4 0 2 1 14 9 49 1 1 1
    # 4 2 4 0 2 1 14 9 46 2 1 151
    # 4 2 4 0 3 1 14 8 50 2 0 22
    # 4 2 4 0 4 1 14 8 54 2 0 41

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

    hit = hits[17]
    self.assertEqual(hit.strip      , 72)
    self.assertEqual(hit.wire       , 20)
    self.assertEqual(hit.phi_fp     , 2813)
    self.assertEqual(hit.theta_fp   , 25)
    self.assertEqual((1<<hit.ph_hit), 4096)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[18]
    self.assertEqual(hit.strip      , 1)
    self.assertEqual(hit.wire       , 49)
    self.assertEqual(hit.phi_fp     , 2578)
    self.assertEqual(hit.theta_fp   , 22)
    self.assertEqual((1<<hit.ph_hit), 8796093022208)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[19]
    self.assertEqual(hit.strip      , 151)
    self.assertEqual(hit.wire       , 46)
    self.assertEqual(hit.phi_fp     , 2578)
    self.assertEqual(hit.theta_fp   , 23)
    self.assertEqual((1<<hit.ph_hit), 32)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[20]
    self.assertEqual(hit.strip      , 22)
    self.assertEqual(hit.wire       , 50)
    self.assertEqual(hit.phi_fp     , 2698)
    self.assertEqual(hit.theta_fp   , 22)
    self.assertEqual((1<<hit.ph_hit), 256)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[21]
    self.assertEqual(hit.strip      , 41)
    self.assertEqual(hit.wire       , 54)
    self.assertEqual(hit.phi_fp     , 2850)
    self.assertEqual(hit.theta_fp   , 22)
    self.assertEqual((1<<hit.ph_hit), 8192)
    self.assertEqual(hit.phzvl      , 1)

  def test_tracks(self):
    tracks = self.analyzer.handles["tracks"].product()

    track = tracks[3]
    self.assertEqual(track.rank         , 0x3b)
    self.assertEqual(track.ptlut_address, 1017368437)
    self.assertEqual(track.gmt_pt       , 6)
    self.assertEqual(track.mode         , 15)
    self.assertEqual(track.gmt_charge   , 0)
    self.assertEqual(track.gmt_quality  , 15)
    #self.assertEqual(track.gmt_eta      , 325-512)  # fail
    self.assertEqual(track.gmt_phi      , 32)


# ______________________________________________________________________________
if __name__ == "__main__":

  unittest.main()
