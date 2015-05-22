#!/usr/bin/env python

import FWCore.ParameterSet.Config as cms

testString = '''
*****************************************************************************************************************************
*    Row   * Instance * EventAuxiliary * EventAuxiliary * EventAuxiliary *   mht.obj.pt() *   met.obj.pt() * ht.obj.sumEt() *
*****************************************************************************************************************************
*      240 *        0 *         166699 *            354 *      380002939 * 616.2370605468 * 575.8770751953 * 1527.970361684 *
*      426 *        0 *         165364 *            138 *      164179591 * 737.7218627929 * 689.7571411132 * 716.9541669560 *

etc. etc.

*****************************************************************************************************************************
'''

import re

def scanToVEventRange( lines ):
    pattern = re.compile('^\*\s*\d+\D+(\d+)\D+(\d+)\D+(\d+).*')

    eventRanges = cms.untracked.VEventRange()

    for line in lines:
        match = pattern.match(line)
        if match!=None:
            run = match.group(1)
            lumi = match.group(2)
            event = match.group(3)
            # print run, lumi, event
            eventRanges.append( '%s:%s:%s' % (run, lumi, event))
    return eventRanges
