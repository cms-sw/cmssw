import unittest

from FWLiteAnalyzer import FWLiteAnalyzer


class TestEvent134854523(unittest.TestCase):
  def setUp(self):
    inputFiles = ["Event134854523_out.root"]
    # 3 2 6 2 1 1 14 8 4 3 0 166
    # 3 2 6 0 2 1 15 10 14 1 0 73
    # 3 2 6 0 2 1 13 7 16 3 1 25
    # 3 2 6 0 2 1 11 2 16 3 0 14
    # 3 2 6 0 3 1 15 10 11 1 0 87
    # 3 2 6 0 4 1 14 9 6 1 1 80
    # 4 2 6 2 1 1 12 4 1 6 0 39

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
    self.assertEqual(hit.strip      , 166-128)
    self.assertEqual(hit.wire       , 4)
    self.assertEqual(hit.phi_fp     , 4725)
    self.assertEqual(hit.theta_fp   , 13)
    self.assertEqual((1<<hit.ph_hit), 65536)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[18]
    self.assertEqual(hit.strip      , 73)
    self.assertEqual(hit.wire       , 14)
    self.assertEqual(hit.phi_fp     , 2005)
    self.assertEqual(hit.theta_fp   , 10)
    self.assertEqual((1<<hit.ph_hit), 33554432)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[19]
    self.assertEqual(hit.strip      , 25)
    self.assertEqual(hit.wire       , 16)
    self.assertEqual(hit.phi_fp     , 4787)
    self.assertEqual(hit.theta_fp   , 11)
    self.assertEqual((1<<hit.ph_hit), 137438953472)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[20]
    self.assertEqual(hit.strip      , 14)
    self.assertEqual(hit.wire       , 16)
    self.assertEqual(hit.phi_fp     , 4882)
    self.assertEqual(hit.theta_fp   , 11)
    self.assertEqual((1<<hit.ph_hit), 1099511627776)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[21]
    self.assertEqual(hit.strip      , 87)
    self.assertEqual(hit.wire       , 11)
    self.assertEqual(hit.phi_fp     , 2021)
    self.assertEqual(hit.theta_fp   , 10)
    self.assertEqual((1<<hit.ph_hit), 33554432)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[22]
    self.assertEqual(hit.strip      , 80)
    self.assertEqual(hit.wire       , 6)
    self.assertEqual(hit.phi_fp     , 1967)
    self.assertEqual(hit.theta_fp   , 9)
    self.assertEqual((1<<hit.ph_hit), 8388608)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[23]
    self.assertEqual(hit.strip      , 39)
    self.assertEqual(hit.wire       , 1)
    self.assertEqual(hit.phi_fp     , 4819)
    self.assertEqual(hit.theta_fp   , 50)
    self.assertEqual((1<<hit.ph_hit), 262144)
    self.assertEqual(hit.phzvl      , 1)

  def test_tracks(self):
    tracks = self.analyzer.handles["tracks"].product()

    track = tracks[4]
    self.assertEqual(track.rank         , 0x4b)
    self.assertEqual(track.ptlut_address, 943798032)
    self.assertEqual(track.gmt_pt       , 6)
    self.assertEqual(track.mode         , 7)
    self.assertEqual(track.gmt_charge   , 1)
    self.assertEqual(track.gmt_quality  , 11)
    self.assertEqual(track.gmt_eta      , 299-512)
    self.assertEqual(track.gmt_phi      , 17)

    track = tracks[5]
    self.assertEqual(track.rank         , 0x38)
    self.assertEqual(track.ptlut_address, 206448190)
    self.assertEqual(track.gmt_pt       , 19)
    self.assertEqual(track.mode         , 12)
    self.assertEqual(track.gmt_charge   , 1)
    self.assertEqual(track.gmt_quality  , 8)
    self.assertEqual(track.gmt_eta      , 301-512)
    self.assertEqual(track.gmt_phi      , 90)


# ______________________________________________________________________________
if __name__ == "__main__":

  unittest.main()
