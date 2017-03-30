import unittest

from FWLiteAnalyzer import FWLiteAnalyzer


class TestEvent1687130991(unittest.TestCase):
  def setUp(self):
    inputFiles = ["Event1687130991_out.root"]
    # 2 2 4 2 1 1 14 9 17 2 1 116
    # 3 2 4 2 1 1 14 8 20 1 0 9
    # 3 2 4 0 2 1 14 8 65 2 0 4
    # 3 2 4 0 2 1 14 8 69 3 0 153
    # 3 2 4 0 3 1 14 9 71 2 1 155
    # 3 2 4 0 3 1 14 9 67 3 1 5
    # 3 2 4 0 4 1 14 9 84 2 1 155
    # 3 2 4 0 4 1 14 9 79 3 1 5
    # 6 2 4 1 1 1 14 8 35 1 0 78

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
    self.assertEqual(hit.strip      , 116)
    self.assertEqual(hit.wire       , 17)
    self.assertEqual(hit.phi_fp     , 3788)
    self.assertEqual(hit.theta_fp   , 21)
    self.assertEqual((1<<hit.ph_hit), 16)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[8]
    self.assertEqual(hit.strip      , 9)
    self.assertEqual(hit.wire       , 20)
    self.assertEqual(hit.phi_fp     , 3736)
    self.assertEqual(hit.theta_fp   , 32)
    self.assertEqual((1<<hit.ph_hit), 4194304)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[9]
    self.assertEqual(hit.strip      , 4)
    self.assertEqual(hit.wire       , 65)
    self.assertEqual(hit.phi_fp     , 3758)
    self.assertEqual(hit.theta_fp   , 30)
    self.assertEqual((1<<hit.ph_hit), 2199023255552)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[10]
    self.assertEqual(hit.strip      , 153)
    self.assertEqual(hit.wire       , 69)
    self.assertEqual(hit.phi_fp     , 3766)
    self.assertEqual(hit.theta_fp   , 30)
    self.assertEqual((1<<hit.ph_hit), 32)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[11]
    self.assertEqual(hit.strip      , 155)
    self.assertEqual(hit.wire       , 71)
    self.assertEqual(hit.phi_fp     , 3766)
    self.assertEqual(hit.theta_fp   , 30)
    self.assertEqual((1<<hit.ph_hit), 4398046511104)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[12]
    self.assertEqual(hit.strip      , 5)
    self.assertEqual(hit.wire       , 67)
    self.assertEqual(hit.phi_fp     , 3766)
    self.assertEqual(hit.theta_fp   , 30)
    self.assertEqual((1<<hit.ph_hit), 32)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[13]
    self.assertEqual(hit.strip      , 155)
    self.assertEqual(hit.wire       , 84)
    self.assertEqual(hit.phi_fp     , 3766)
    self.assertEqual(hit.theta_fp   , 30)
    self.assertEqual((1<<hit.ph_hit), 4398046511104)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[14]
    self.assertEqual(hit.strip      , 5)
    self.assertEqual(hit.wire       , 79)
    self.assertEqual(hit.phi_fp     , 3766)
    self.assertEqual(hit.theta_fp   , 30)
    self.assertEqual((1<<hit.ph_hit), 32)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[15]
    self.assertEqual(hit.strip      , 78)
    self.assertEqual(hit.wire       , 35)
    self.assertEqual(hit.phi_fp     , 1584)
    self.assertEqual(hit.theta_fp   , 38)
    self.assertEqual((1<<hit.ph_hit), 2048)
    self.assertEqual(hit.phzvl      , 1)

  def test_tracks(self):
    tracks = self.analyzer.handles["tracks"].product()

    track = tracks[1]
    self.assertEqual(track.rank         , 0x6b)
    self.assertEqual(track.ptlut_address, 1023149078)
    self.assertEqual(track.gmt_pt       , 102)
    self.assertEqual(track.mode         , 15)
    self.assertEqual(track.gmt_charge   , 1)
    self.assertEqual(track.gmt_quality  , 15)
    self.assertEqual(track.gmt_eta      , 337-512)
    self.assertEqual(track.gmt_phi      , 63)

    #track = tracks[2]
    #self.assertEqual(track.rank         , 0x4b)
    #self.assertEqual(track.ptlut_address, 954589184)
    #self.assertEqual(track.gmt_pt       , 157)
    #self.assertEqual(track.mode         , 7)
    #self.assertEqual(track.gmt_charge   , 1)
    #self.assertEqual(track.gmt_quality  , 11)
    #self.assertEqual(track.gmt_eta      , 337-512)
    #self.assertEqual(track.gmt_phi      , 63)
    #self.assertEqual(track.sector       , 4)

# ______________________________________________________________________________
if __name__ == "__main__":

  unittest.main()
