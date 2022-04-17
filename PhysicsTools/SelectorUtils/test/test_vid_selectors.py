import unittest

class TestVIDSelectros(unittest.TestCase):

    def test_vid_selectors(self):
        # Test that all VID selectors for Python/FWLite can be instantiated

        from RecoMuon.MuonIdentification.VIDMuonSelector import VIDMuonSelector
        from RecoEgamma.ElectronIdentification.VIDElectronSelector import VIDElectronSelector
        from RecoEgamma.PhotonIdentification.VIDPhotonSelector import VIDPhotonSelector

        VIDMuonSelector()
        VIDElectronSelector()
        VIDPhotonSelector()


if __name__ == "__main__":

    unittest.main(verbosity=2)
