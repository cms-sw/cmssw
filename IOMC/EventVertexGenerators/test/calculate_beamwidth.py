#!/usr/bin/env python
############################################################################
#
# Author: Francisco Yumiceva
# yumiceva@fnal.gov
#
# Fermilab, 2009
#
############################################################################

"""
given beta-star and beam energy estimate: gamma, emittance, and beam width
at Z = 0

"""

import math

beam = input("beam energy in GeV: ")
betastar = input("beta-star in m: ")
normemittance = input("normalized emittance in m: ")

gamma = beam/0.9383

emittance = normemittance/(gamma*math.sqrt(1.-1/(gamma*gamma)))

width = math.sqrt(emittance*betastar)

print " emittance in m: " + str(emittance)
print " beam width in m: " + str(width)

