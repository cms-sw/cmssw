import unittest

from FWLiteAnalyzer import FWLiteAnalyzer


class TestEvent300921221(unittest.TestCase):
  def setUp(self):
    inputFiles = ["Event300921221_out.root"]
    # 3 1 2 1 1 1 15 10 6 2 0 188
    # 3 1 2 1 1 1 14 8 6 2 0 218
    # 3 1 2 1 1 1 15 10 4 2 0 188
    # 3 1 2 1 1 1 14 8 4 2 0 218
    # 3 1 2 0 2 1 15 10 28 1 0 133
    # 3 1 2 0 2 1 14 9 28 1 1 148
    # 3 1 2 0 2 1 15 10 20 1 0 133
    # 3 1 2 0 2 1 14 9 20 1 1 148
    # 3 1 2 0 3 1 15 10 28 1 0 24
    # 3 1 2 0 3 1 15 10 28 1 0 11
    # 3 1 2 0 3 1 15 10 19 1 0 24
    # 3 1 2 0 3 1 15 10 19 1 0 11
    # 3 1 2 0 4 1 15 10 28 1 0 12
    # 3 1 2 0 4 1 15 10 28 1 0 24
    # 3 1 2 0 4 1 15 10 18 1 0 12
    # 3 1 2 0 4 1 15 10 18 1 0 24

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

    hit = hits[3]
    self.assertEqual(hit.strip      , 188-128)
    self.assertEqual(hit.wire       , 6)
    self.assertEqual(hit.phi_fp     , 2331)
    self.assertEqual(hit.theta_fp   , 14)
    self.assertEqual((1<<hit.ph_hit), 65536)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[4]
    self.assertEqual(hit.strip      , 218-128)
    self.assertEqual(hit.wire       , 6)
    self.assertEqual(hit.phi_fp     , 2529)
    self.assertEqual(hit.theta_fp   , 15)
    self.assertEqual((1<<hit.ph_hit), 4194304)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[5]
    self.assertEqual(hit.strip      , 188-128)
    self.assertEqual(hit.wire       , 4)
    self.assertEqual(hit.phi_fp     , 2331)
    self.assertEqual(hit.theta_fp   , 12)
    self.assertEqual((1<<hit.ph_hit), 65536)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[6]
    self.assertEqual(hit.strip      , 218-128)
    self.assertEqual(hit.wire       , 4)
    self.assertEqual(hit.phi_fp     , 2529)
    self.assertEqual(hit.theta_fp   , 13)
    self.assertEqual((1<<hit.ph_hit), 4194304)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[7]
    self.assertEqual(hit.strip      , 133)
    self.assertEqual(hit.wire       , 28)
    self.assertEqual(hit.phi_fp     , 2380)
    self.assertEqual(hit.theta_fp   , 15)
    self.assertEqual((1<<hit.ph_hit), 68719476736)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[8]
    self.assertEqual(hit.strip      , 148)
    self.assertEqual(hit.wire       , 28)
    self.assertEqual(hit.phi_fp     , 2502)
    self.assertEqual(hit.theta_fp   , 15)
    self.assertEqual((1<<hit.ph_hit), 1099511627776)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[9]
    self.assertEqual(hit.strip      , 133)
    self.assertEqual(hit.wire       , 20)
    self.assertEqual(hit.phi_fp     , 2380)
    self.assertEqual(hit.theta_fp   , 12)
    self.assertEqual((1<<hit.ph_hit), 68719476736)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[10]
    self.assertEqual(hit.strip      , 148)
    self.assertEqual(hit.wire       , 20)
    self.assertEqual(hit.phi_fp     , 2502)
    self.assertEqual(hit.theta_fp   , 12)
    self.assertEqual((1<<hit.ph_hit), 1099511627776)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[11]
    self.assertEqual(hit.strip      , 24)
    self.assertEqual(hit.wire       , 28)
    self.assertEqual(hit.phi_fp     , 2388)
    self.assertEqual(hit.theta_fp   , 15)
    self.assertEqual((1<<hit.ph_hit), 137438953472)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[12]
    self.assertEqual(hit.strip      , 11)
    self.assertEqual(hit.wire       , 28)
    self.assertEqual(hit.phi_fp     , 2492)
    self.assertEqual(hit.theta_fp   , 15)
    self.assertEqual((1<<hit.ph_hit), 1099511627776)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[13]
    self.assertEqual(hit.strip      , 24)
    self.assertEqual(hit.wire       , 19)
    self.assertEqual(hit.phi_fp     , 2388)
    self.assertEqual(hit.theta_fp   , 12)
    self.assertEqual((1<<hit.ph_hit), 137438953472)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[14]
    self.assertEqual(hit.strip      , 11)
    self.assertEqual(hit.wire       , 19)
    self.assertEqual(hit.phi_fp     , 2492)
    self.assertEqual(hit.theta_fp   , 12)
    self.assertEqual((1<<hit.ph_hit), 1099511627776)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[15]
    self.assertEqual(hit.strip      , 12)
    self.assertEqual(hit.wire       , 28)
    self.assertEqual(hit.phi_fp     , 2484)
    self.assertEqual(hit.theta_fp   , 15)
    self.assertEqual((1<<hit.ph_hit), 1099511627776)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[16]
    self.assertEqual(hit.strip      , 24)
    self.assertEqual(hit.wire       , 28)
    self.assertEqual(hit.phi_fp     , 2388)
    self.assertEqual(hit.theta_fp   , 15)
    self.assertEqual((1<<hit.ph_hit), 137438953472)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[17]
    self.assertEqual(hit.strip      , 12)
    self.assertEqual(hit.wire       , 18)
    self.assertEqual(hit.phi_fp     , 2484)
    self.assertEqual(hit.theta_fp   , 12)
    self.assertEqual((1<<hit.ph_hit), 1099511627776)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[18]
    self.assertEqual(hit.strip      , 24)
    self.assertEqual(hit.wire       , 18)
    self.assertEqual(hit.phi_fp     , 2388)
    self.assertEqual(hit.theta_fp   , 12)
    self.assertEqual((1<<hit.ph_hit), 137438953472)
    self.assertEqual(hit.phzvl      , 1)


  def test_tracks(self):
    tracks = self.analyzer.handles["tracks"].product()

    track = tracks[0]
    self.assertEqual(track.rank         , 0x6b)
    self.assertEqual(track.ptlut_address, 1014793499)
    self.assertEqual(track.gmt_pt       , 55)
    self.assertEqual(track.mode         , 15)
    self.assertEqual(track.gmt_charge   , 0)
    self.assertEqual(track.gmt_quality  , 15)
    self.assertEqual(track.gmt_eta      , 201)
    self.assertEqual(track.gmt_phi      , 30)

    track = tracks[1]
    self.assertEqual(track.rank         , 0x3f)
    self.assertEqual(track.ptlut_address, 1014760497)
    self.assertEqual(track.gmt_pt       , 32)
    self.assertEqual(track.mode         , 15)
    self.assertEqual(track.gmt_charge   , 1)
    self.assertEqual(track.gmt_quality  , 15)
    self.assertEqual(track.gmt_eta      , 207)
    self.assertEqual(track.gmt_phi      , 27)


# ______________________________________________________________________________
if __name__ == "__main__":

  unittest.main()
