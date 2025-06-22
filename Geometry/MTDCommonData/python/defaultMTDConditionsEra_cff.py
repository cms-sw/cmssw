import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _centralgeo

MTD_DEFAULT_VERSION = "Run4D121"

def check_mtdgeo():
    if MTD_DEFAULT_VERSION != _centralgeo.DEFAULT_VERSION:
        print("MTD test geometry scenario ",MTD_DEFAULT_VERSION," different than CMS default ",_centralgeo.DEFAULT_VERSION)
