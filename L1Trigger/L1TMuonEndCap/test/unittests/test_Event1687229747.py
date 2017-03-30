import unittest

from FWLiteAnalyzer import FWLiteAnalyzer


class TestEvent1687229747(unittest.TestCase):
  def setUp(self):
    inputFiles = ["Event1687229747_out.root"]
    # 3 2 1 2 1 1 15 10 11 2 0 184
    # 3 2 1 0 2 1 15 10 27 3 0 122
    # 3 2 1 0 3 1 15 10 34 3 0 37
    # 3 2 1 0 4 1 15 10 34 3 0 37
    # 3 2 1 0 4 1 15 10 40 3 0 37
    # 4 2 1 0 2 1 15 10 27 3 0 122
    # 4 2 1 0 3 1 15 10 34 3 0 37
    # 4 2 1 0 4 1 15 10 34 3 0 37
    # 4 2 1 0 4 1 15 10 40 3 0 37

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

    hit = hits[1]
    self.assertEqual(hit.strip      , 184-128)
    self.assertEqual(hit.wire       , 11)
    self.assertEqual(hit.phi_fp     , 4000)
    self.assertEqual(hit.theta_fp   , 17)
    self.assertEqual((1<<hit.ph_hit), 2048)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[2]
    self.assertEqual(hit.strip      , 122)
    self.assertEqual(hit.wire       , 27)
    self.assertEqual(hit.phi_fp     , 4012)
    self.assertEqual(hit.theta_fp   , 16)
    self.assertEqual((1<<hit.ph_hit), 4096)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[3]
    self.assertEqual(hit.strip      , 37)
    self.assertEqual(hit.wire       , 34)
    self.assertEqual(hit.phi_fp     , 4020)
    self.assertEqual(hit.theta_fp   , 16)
    self.assertEqual((1<<hit.ph_hit), 8192)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[4]
    self.assertEqual(hit.strip      , 37)
    self.assertEqual(hit.wire       , 34)
    self.assertEqual(hit.phi_fp     , 4020)
    self.assertEqual(hit.theta_fp   , 16)
    self.assertEqual((1<<hit.ph_hit), 8192)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[5]
    self.assertEqual(hit.strip      , 37)
    self.assertEqual(hit.wire       , 40)
    self.assertEqual(hit.phi_fp     , 4020)
    self.assertEqual(hit.theta_fp   , 18)
    self.assertEqual((1<<hit.ph_hit), 8192)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[6]
    self.assertEqual(hit.strip      , 122)
    self.assertEqual(hit.wire       , 27)
    self.assertEqual(hit.phi_fp     , 4012)
    self.assertEqual(hit.theta_fp   , 16)
    self.assertEqual((1<<hit.ph_hit), 4096)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[7]
    self.assertEqual(hit.strip      , 37)
    self.assertEqual(hit.wire       , 34)
    self.assertEqual(hit.phi_fp     , 4020)
    self.assertEqual(hit.theta_fp   , 16)
    self.assertEqual((1<<hit.ph_hit), 8192)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[8]
    self.assertEqual(hit.strip      , 37)
    self.assertEqual(hit.wire       , 34)
    self.assertEqual(hit.phi_fp     , 4020)
    self.assertEqual(hit.theta_fp   , 16)
    self.assertEqual((1<<hit.ph_hit), 8192)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[9]
    self.assertEqual(hit.strip      , 37)
    self.assertEqual(hit.wire       , 40)
    self.assertEqual(hit.phi_fp     , 4020)
    self.assertEqual(hit.theta_fp   , 18)
    self.assertEqual((1<<hit.ph_hit), 8192)
    self.assertEqual(hit.phzvl      , 1)

  def test_tracks(self):
    tracks = self.analyzer.handles["tracks"].product()

    track = tracks[0]
    self.assertEqual(track.rank         , 0x6b)
    self.assertEqual(track.ptlut_address, 1015809036)
    self.assertEqual(track.gmt_pt       , 140)
    self.assertEqual(track.mode         , 15)
    self.assertEqual(track.gmt_charge   , 1)
    self.assertEqual(track.gmt_quality  , 15)
    self.assertEqual(track.gmt_eta      , 312-512)
    self.assertEqual(track.gmt_phi      , 69)


# ______________________________________________________________________________
if __name__ == "__main__":

  unittest.main()
