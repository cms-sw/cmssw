#!/usr/bin/env python3
from FWCore.ParameterSet.Types import PSet
import FWCore.ParameterSet.Config as cms

class DTDQMConfig( PSet ):

  # constructor

  def __init__( self ):
    PSet.__init__( self )       
    self.__processAB7Digis = False
    self.__processAB7TPs   = False
    self.__runWithLargeTB  = False
    self.__tbTDCPedestal   = 0
  
  # getters
  
  def getProcessAB7Digis( self ):
    return self.__processAB7Digis
    
  def getProcessAB7TPs( self ):
    return self.__processAB7TPs

  def getRunWithLargeTB( self ):
    return self.__runWithLargeTB

  def getTBTDCPedestal( self ):
    return self.__tbTDCPedestal
    
  # setters

  def setProcessAB7Digis( self, processAB7Digis ):
      self.__processAB7Digis = processAB7Digis
    
  def setProcessAB7TPs( self, processAB7TPs ):
      self.__processAB7TPs = processAB7TPs
    
  def setRunWithLargeTB( self,runWithLargeTB ):
      self.__runWithLargeTB = runWithLargeTB

  def setTBTDCPedestal( self, tbTDCPedestal ):
      self.__tbTDCPedestal = tbTDCPedestal
    
  # str and repr

  def __str__( self ):
    return "DTDQMConfig: processAB7Digis='%r' processAB7TPs='%r' runWithLargeTB='%r' tbTDCPedestal='%d'" % (self.__processAB7Digis, \
                                                                                                            self.__processAB7TPs,   \
                                                                                                            self.__runWithLargeTB,  \
                                                                                                            self.__tbTDCPedestal)
    
  def __repr__( self ):
    return "DTDQMConfig: processAB7Digis='%r' processAB7TPs='%r' runWithLargeTB='%r' tbTDCPedestal='%d'" % (self.__processAB7Digis, \
                                                                                                            self.__processAB7TPs,   \
                                                                                                            self.__runWithLargeTB,  \
                                                                                                            self.__tbTDCPedestal)
