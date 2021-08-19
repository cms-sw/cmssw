import FWCore.ParameterSet.Config as cms

from CalibPPS.ESProducers.ppsAssociationCutsESSource_cfi import *

p2016 = cms.PSet(
    validityRange=cms.EventRange("273725:min - 284044:max"),
    association_cuts_45=cms.PSet(
        xi_cut_mean =cms.string("0."),
        xi_cut_threshold=cms.string("0.010"),

        ti_tr_min=cms.double(-1.5),
        ti_tr_max=cms.double(2.0)
    ),
    association_cuts_56=cms.PSet(
        xi_cut_mean= cms.string("0."),
        xi_cut_threshold=cms.string("0.015"),

        ti_tr_min=cms.double(-1.5),
        ti_tr_max=cms.double(2.0)
    ),
)
ppsAssociationCutsESSource.configuration.append(p2016)

p2017 = cms.PSet(
    validityRange=cms.EventRange("297046:min - 307082:max"),
    association_cuts_45=cms.PSet(
        xi_cut_mean=cms.string("+6.0695e-5"),
        xi_cut_threshold=cms.string("5. * 0.00121"),
        ti_tr_min=cms.double(-1.5),
        ti_tr_max=cms.double(2.0)
    ),
    association_cuts_56=cms.PSet(
        y_cut_mean=cms.string("-0.022612"),
        y_cut_threshold=cms.string("5. * 0.14777"),
        xi_cut_mean=cms.string("+8.012857e-5"),
        xi_cut_threshold=cms.string("5. * 0.0020627"),
        ti_tr_min=cms.double(-1.5),
        ti_tr_max=cms.double(2.0)
    ),
)
ppsAssociationCutsESSource.configuration.append(p2017)

p2018 = cms.PSet(
    validityRange=cms.EventRange("314747:min - 325175:max"),
    association_cuts_45=cms.PSet(
        x_cut_mean=cms.string("-0.065194856"),
        x_cut_threshold=cms.string("4. * 0.16008188"),
        y_cut_mean=cms.string("+0.10973631"),
        y_cut_threshold=cms.string("4. * 0.1407986"),
        xi_cut_mean=cms.string("+3.113062e-5"),
        xi_cut_threshold=cms.string("4. * 0.0012403586"),
        ti_tr_min=cms.double(-1.5),
        ti_tr_max=cms.double(2.0)
    ),
    association_cuts_56=cms.PSet(
        x_cut_mean=cms.string("+0.073016431"),
        x_cut_threshold=cms.string("5. * 0.18126434"),
        y_cut_mean=cms.string("+0.064261029"),
        y_cut_threshold=cms.string("5. * 0.14990802"),
        xi_cut_mean=cms.string("-1.1852528e-5"),
        xi_cut_threshold=cms.string("5. * 0.002046409"),
        ti_tr_min=cms.double(-1.5),
        ti_tr_max=cms.double(2.0)
    ),
)
ppsAssociationCutsESSource.configuration.append(p2018)

pFuture = cms.PSet(
    validityRange=cms.EventRange("343890:min - 999999:max"),
    association_cuts_45=cms.PSet(
        x_cut_mean=cms.string("-0.065194856"),
        x_cut_threshold=cms.string("4. * 0.16008188"),
        y_cut_mean=cms.string("+0.10973631"),
        y_cut_threshold=cms.string("4. * 0.1407986"),
        xi_cut_mean=cms.string("+3.113062e-5"),
        xi_cut_threshold=cms.string("4. * 0.0012403586"),
        ti_tr_min=cms.double(-1.5),
        ti_tr_max=cms.double(2.0)
    ),
    association_cuts_56=cms.PSet(
        x_cut_mean=cms.string("+0.073016431"),
        x_cut_threshold=cms.string("5. * 0.18126434"),
        y_cut_mean=cms.string("+0.064261029"),
        y_cut_threshold=cms.string("5. * 0.14990802"),
        xi_cut_mean=cms.string("-1.1852528e-5"),
        xi_cut_threshold=cms.string("5. * 0.002046409"),
        ti_tr_min=cms.double(-1.5),
        ti_tr_max=cms.double(2.0)
    ),
)
ppsAssociationCutsESSource.configuration.append(pFuture)

def use_single_infinite_iov_entry(ppsAssociationCutsESSource, iov):
    ppsAssociationCutsESSource.configuration = cms.VPSet()
    iov.validityRange = cms.EventRange("0:min - 999999:max")
    ppsAssociationCutsESSource.configuration.append(iov)
