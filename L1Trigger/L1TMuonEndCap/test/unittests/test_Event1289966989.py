import unittest

from FWLiteAnalyzer import FWLiteAnalyzer


class TestEvent1289966989(unittest.TestCase):
  def setUp(self):
    inputFiles = ["Event1289966989_out.root"]
    # 3 1 5 2 1 1 14 8 9 2 0 190
    # 3 1 5 2 1 1 15 10 3 2 0 205
    # 3 1 5 0 2 1 15 10 26 3 0 68
    # 3 1 5 0 2 1 15 10 11 3 0 43
    # 3 1 5 0 3 1 15 10 15 3 0 90
    # 3 1 5 0 3 1 15 10 33 3 0 119
    # 3 1 5 0 4 1 15 10 34 3 0 119
    # 3 1 5 0 4 1 15 10 14 3 0 90

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

    hit = hits[2]
    self.assertEqual(hit.strip      , 190-128)
    self.assertEqual(hit.wire       , 9)
    self.assertEqual(hit.phi_fp     , 4142)
    self.assertEqual(hit.theta_fp   , 15)
    self.assertEqual((1<<hit.ph_hit), 32768)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[3]
    self.assertEqual(hit.strip      , 205-128)
    self.assertEqual(hit.wire       , 9)
    self.assertEqual(hit.phi_fp     , 4244)
    self.assertEqual(hit.theta_fp   , 15)
    self.assertEqual((1<<hit.ph_hit), 524288)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[4]
    self.assertEqual(hit.strip      , 190-128)
    self.assertEqual(hit.wire       , 3)
    self.assertEqual(hit.phi_fp     , 4142)
    self.assertEqual(hit.theta_fp   , 9)
    self.assertEqual((1<<hit.ph_hit), 32768)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[5]
    self.assertEqual(hit.strip      , 205-128)
    self.assertEqual(hit.wire       , 3)
    self.assertEqual(hit.phi_fp     , 4244)
    self.assertEqual(hit.theta_fp   , 9)
    self.assertEqual((1<<hit.ph_hit), 524288)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[6]
    self.assertEqual(hit.strip      , 68)
    self.assertEqual(hit.wire       , 26)
    self.assertEqual(hit.phi_fp     , 4260)
    self.assertEqual(hit.theta_fp   , 16)
    self.assertEqual((1<<hit.ph_hit), 1048576)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[7]
    self.assertEqual(hit.strip      , 43)
    self.assertEqual(hit.wire       , 26)
    self.assertEqual(hit.phi_fp     , 4060)
    self.assertEqual(hit.theta_fp   , 16)
    self.assertEqual((1<<hit.ph_hit), 16384)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[8]
    self.assertEqual(hit.strip      , 68)
    self.assertEqual(hit.wire       , 11)
    self.assertEqual(hit.phi_fp     , 4260)
    self.assertEqual(hit.theta_fp   , 10)
    self.assertEqual((1<<hit.ph_hit), 1048576)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[9]
    self.assertEqual(hit.strip      , 43)
    self.assertEqual(hit.wire       , 11)
    self.assertEqual(hit.phi_fp     , 4060)
    self.assertEqual(hit.theta_fp   , 10)
    self.assertEqual((1<<hit.ph_hit), 16384)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[10]
    self.assertEqual(hit.strip      , 90)
    self.assertEqual(hit.wire       , 15)
    self.assertEqual(hit.phi_fp     , 4260)
    self.assertEqual(hit.theta_fp   , 10)
    self.assertEqual((1<<hit.ph_hit), 1048576)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[11]
    self.assertEqual(hit.strip      , 119)
    self.assertEqual(hit.wire       , 15)
    self.assertEqual(hit.phi_fp     , 4028)
    self.assertEqual(hit.theta_fp   , 10)
    self.assertEqual((1<<hit.ph_hit), 8192)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[12]
    self.assertEqual(hit.strip      , 90)
    self.assertEqual(hit.wire       , 33)
    self.assertEqual(hit.phi_fp     , 4260)
    self.assertEqual(hit.theta_fp   , 16)
    self.assertEqual((1<<hit.ph_hit), 1048576)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[13]
    self.assertEqual(hit.strip      , 119)
    self.assertEqual(hit.wire       , 33)
    self.assertEqual(hit.phi_fp     , 4028)
    self.assertEqual(hit.theta_fp   , 16)
    self.assertEqual((1<<hit.ph_hit), 8192)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[14]
    self.assertEqual(hit.strip      , 119)
    self.assertEqual(hit.wire       , 34)
    self.assertEqual(hit.phi_fp     , 4028)
    self.assertEqual(hit.theta_fp   , 16)
    self.assertEqual((1<<hit.ph_hit), 8192)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[15]
    self.assertEqual(hit.strip      , 90)
    self.assertEqual(hit.wire       , 34)
    self.assertEqual(hit.phi_fp     , 4260)
    self.assertEqual(hit.theta_fp   , 16)
    self.assertEqual((1<<hit.ph_hit), 1048576)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[16]
    self.assertEqual(hit.strip      , 119)
    self.assertEqual(hit.wire       , 14)
    self.assertEqual(hit.phi_fp     , 4028)
    self.assertEqual(hit.theta_fp   , 10)
    self.assertEqual((1<<hit.ph_hit), 8192)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[17]
    self.assertEqual(hit.strip      , 90)
    self.assertEqual(hit.wire       , 14)
    self.assertEqual(hit.phi_fp     , 4260)
    self.assertEqual(hit.theta_fp   , 10)
    self.assertEqual((1<<hit.ph_hit), 1048576)
    self.assertEqual(hit.phzvl      , 1)


  def test_tracks(self):
    tracks = self.analyzer.handles["tracks"].product()

    track = tracks[0]
    self.assertEqual(track.rank         , 0x6b)
    self.assertEqual(track.ptlut_address, 1011613712)
    self.assertEqual(track.gmt_pt       , 106)
    self.assertEqual(track.mode         , 15)
    self.assertEqual(track.gmt_charge   , 1)
    self.assertEqual(track.gmt_quality  , 15)
    self.assertEqual(track.gmt_eta      , 212)
    self.assertEqual(track.gmt_phi      , 76)

    track = tracks[1]
    self.assertEqual(track.rank         , 0x3b)
    self.assertEqual(track.ptlut_address, 1011092559)
    self.assertEqual(track.gmt_pt       , 16)
    self.assertEqual(track.mode         , 15)
    self.assertEqual(track.gmt_charge   , 0)
    self.assertEqual(track.gmt_quality  , 15)
    self.assertEqual(track.gmt_eta      , 212)
    self.assertEqual(track.gmt_phi      , 71)


# ______________________________________________________________________________
if __name__ == "__main__":

  unittest.main()
