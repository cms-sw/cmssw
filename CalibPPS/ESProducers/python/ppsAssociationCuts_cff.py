import FWCore.ParameterSet.Config as cms

from CalibPPS.ESProducers.ppsAssociationCutsESSource_cfi import *

p2016 = cms.PSet(
    validityRange=cms.EventRange("273725:min - 284044:max"),
    association_cuts_45=cms.PSet(
        x_cut_apply=cms.bool(False),
        y_cut_apply=cms.bool(False),
        xi_cut_apply=cms.bool(True),
        xi_cut_value=cms.double(0.010),
        th_y_cut_apply=cms.bool(False),
        ti_tr_min=cms.double(-1.5),
        ti_tr_max=cms.double(2.0)
    ),
    association_cuts_56=cms.PSet(
        x_cut_apply=cms.bool(False),
        y_cut_apply=cms.bool(False),
        xi_cut_apply=cms.bool(True),
        xi_cut_value=cms.double(0.015),
        th_y_cut_apply=cms.bool(False),
        ti_tr_min=cms.double(-1.5),
        ti_tr_max=cms.double(2.0)
    ),
)
ppsAssociationCutsESSource.configuration.append(p2016)

p2017 = cms.PSet(
    validityRange=cms.EventRange("297046:min - 307082:max"),
    association_cuts_45=cms.PSet(
        x_cut_apply=cms.bool(False),
        y_cut_apply=cms.bool(False),
        xi_cut_apply=cms.bool(True),
        xi_cut_value=cms.double(5. * 0.00121),
        xi_cut_mean=cms.double(+6.0695e-5),
        th_y_cut_apply=cms.bool(False),
        ti_tr_min=cms.double(-1.5),
        ti_tr_max=cms.double(2.0)
    ),
    association_cuts_56=cms.PSet(
        x_cut_apply=cms.bool(False),
        y_cut_apply=cms.bool(True),
        y_cut_value=cms.double(5. * 0.14777),
        y_cut_mean=cms.double(-0.022612),
        xi_cut_apply=cms.bool(True),
        xi_cut_value=cms.double(5. * 0.0020627),
        xi_cut_mean=cms.double(+8.012857e-5),
        th_y_cut_apply=cms.bool(False),
        ti_tr_min=cms.double(-1.5),
        ti_tr_max=cms.double(2.0)
    ),
)
ppsAssociationCutsESSource.configuration.append(p2017)

p2018 = cms.PSet(
    validityRange=cms.EventRange("314747:min - 325175:max"),
    association_cuts_45=cms.PSet(
        x_cut_apply=cms.bool(True),
        x_cut_value=cms.double(4. * 0.16008188),
        x_cut_mean=cms.double(-0.065194856),
        y_cut_apply=cms.bool(True),
        y_cut_value=cms.double(4. * 0.1407986),
        y_cut_mean=cms.double(+0.10973631),
        xi_cut_apply=cms.bool(True),
        xi_cut_value=cms.double(4. * 0.0012403586),
        xi_cut_mean=cms.double(+3.113062e-5),
        th_y_cut_apply=cms.bool(False),
        ti_tr_min=cms.double(-1.5),
        ti_tr_max=cms.double(2.0)
    ),
    association_cuts_56=cms.PSet(
        x_cut_apply=cms.bool(True),
        x_cut_value=cms.double(5. * 0.18126434),
        x_cut_mean=cms.double(+0.073016431),
        y_cut_apply=cms.bool(True),
        y_cut_value=cms.double(5. * 0.14990802),
        y_cut_mean=cms.double(+0.064261029),
        xi_cut_apply=cms.bool(True),
        xi_cut_value=cms.double(5. * 0.002046409),
        xi_cut_mean=cms.double(-1.1852528e-5),
        th_y_cut_apply=cms.bool(False),
        ti_tr_min=cms.double(-1.5),
        ti_tr_max=cms.double(2.0)
    ),
)
ppsAssociationCutsESSource.configuration.append(p2018)

pFuture = cms.PSet(
    validityRange=cms.EventRange("343890:min - 999999:max"),
    association_cuts_45=cms.PSet(
        x_cut_apply=cms.bool(True),
        x_cut_value=cms.double(4. * 0.16008188),
        x_cut_mean=cms.double(-0.065194856),
        y_cut_apply=cms.bool(True),
        y_cut_value=cms.double(4. * 0.1407986),
        y_cut_mean=cms.double(+0.10973631),
        xi_cut_apply=cms.bool(True),
        xi_cut_value=cms.double(4. * 0.0012403586),
        xi_cut_mean=cms.double(+3.113062e-5),
        th_y_cut_apply=cms.bool(False),
        ti_tr_min=cms.double(-1.5),
        ti_tr_max=cms.double(2.0)
    ),
    association_cuts_56=cms.PSet(
        x_cut_apply=cms.bool(True),
        x_cut_value=cms.double(5. * 0.18126434),
        x_cut_mean=cms.double(+0.073016431),
        y_cut_apply=cms.bool(True),
        y_cut_value=cms.double(5. * 0.14990802),
        y_cut_mean=cms.double(+0.064261029),
        xi_cut_apply=cms.bool(True),
        xi_cut_value=cms.double(5. * 0.002046409),
        xi_cut_mean=cms.double(-1.1852528e-5),
        th_y_cut_apply=cms.bool(False),
        ti_tr_min=cms.double(-1.5),
        ti_tr_max=cms.double(2.0)
    ),
)
ppsAssociationCutsESSource.configuration.append(pFuture)

def use_single_infinite_iov_entry(ppsAssociationCutsESSource, iov):
    ppsAssociationCutsESSource.configuration = cms.VPSet()
    iov.validityRange = cms.EventRange("0:min - 999999:max")
    ppsAssociationCutsESSource.configuration.append(iov)
