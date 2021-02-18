#!/bin/sh -e

cmsRun ${CMSSW_BASE}/src/GeneratorInterface/RivetInterface/test/rivet_cfg.py
! grep "nan" out.yoda
