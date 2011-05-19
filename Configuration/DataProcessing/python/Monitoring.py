#!/usr/bin/env python
# encoding: utf-8
"""
Monitoring.py

Created by Dave Evans on 2011-05-19.
Copyright (c) 2011 Fermilab. All rights reserved.
"""

import FWCore.ParameterSet.Config as cms


def addMonitoring(process):
    """
    _addMonitoring_
    
    Add the monitoring services to the process provided
    in order to write out performance summaries to the framework job report
    """

    process.add_(cms.Service("SimpleMemoryCheck"))
    process.add_(cms.Service("Timing"))
    process.Timing.summaryOnly = cms.untracked(cms.bool(True))



