import unittest

from FWLiteAnalyzer import FWLiteAnalyzer


class TestEvent1687278667(unittest.TestCase):
  def setUp(self):
    inputFiles = ["Event1687278667_out.root"]
    # 1 1 2 2 5 1 14 9 19 1 1 29
    # 1 1 2 0 5 1 15 10 40 4 0 134
    # 3 1 2 2 1 1 12 5 22 1 1 91
    # 3 1 2 2 1 1 12 5 6 2 1 215
    # 3 1 2 0 2 1 15 10 62 3 0 41
    # 3 1 2 0 3 1 15 10 55 3 0 122
    # 3 1 2 0 4 1 14 8 59 3 0 139

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
    self.assertEqual(hit.strip      , 29)
    self.assertEqual(hit.wire       , 19)
    self.assertEqual(hit.phi_fp     , 873)
    self.assertEqual(hit.theta_fp   , 26)
    self.assertEqual((1<<hit.ph_hit), 64)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[3]
    self.assertEqual(hit.strip      , 134)
    self.assertEqual(hit.wire       , 40)
    self.assertEqual(hit.phi_fp     , 1188)
    self.assertEqual(hit.theta_fp   , 21)
    self.assertEqual((1<<hit.ph_hit), 68719476736)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[4]
    self.assertEqual(hit.strip      , 91)
    self.assertEqual(hit.wire       , 22)
    self.assertEqual(hit.phi_fp     , 3588)
    self.assertEqual(hit.theta_fp   , 32)
    self.assertEqual((1<<hit.ph_hit), 131072)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[5]
    self.assertEqual(hit.strip      , 215-128)
    self.assertEqual(hit.wire       , 6)
    self.assertEqual(hit.phi_fp     , 4314)
    self.assertEqual(hit.theta_fp   , 13)
    self.assertEqual((1<<hit.ph_hit), 2097152)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[6]
    self.assertEqual(hit.strip      , 41)
    self.assertEqual(hit.wire       , 62)
    self.assertEqual(hit.phi_fp     , 4043)
    self.assertEqual(hit.theta_fp   , 27)
    self.assertEqual((1<<hit.ph_hit), 8192)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[7]
    self.assertEqual(hit.strip      , 122)
    self.assertEqual(hit.wire       , 55)
    self.assertEqual(hit.phi_fp     , 4004)
    self.assertEqual(hit.theta_fp   , 25)
    self.assertEqual((1<<hit.ph_hit), 4096)
    self.assertEqual(hit.phzvl      , 1)

    hit = hits[8]
    self.assertEqual(hit.strip      , 139)
    self.assertEqual(hit.wire       , 59)
    self.assertEqual(hit.phi_fp     , 3869)
    self.assertEqual(hit.theta_fp   , 24)
    self.assertEqual((1<<hit.ph_hit), 256)
    self.assertEqual(hit.phzvl      , 1)

  def test_tracks(self):
    tracks = self.analyzer.handles["tracks"].product()

    track = tracks[0]
    self.assertEqual(track.rank         , 0x0f)
    self.assertEqual(track.ptlut_address, 952114599)
    self.assertEqual(track.gmt_pt       , 6)
    self.assertEqual(track.mode         , 7)
    self.assertEqual(track.gmt_charge   , 0)
    self.assertEqual(track.gmt_quality  , 11)
    self.assertEqual(track.gmt_eta      , 179)
    self.assertEqual(track.gmt_phi      , 70)


# ______________________________________________________________________________
if __name__ == "__main__":

  unittest.main()
