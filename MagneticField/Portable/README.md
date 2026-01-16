
 # MagneticField/Portable 

This package will host simple and efficient implementations of the magnetic field map for use in well-defined scenarios on GPUs and other heterogeneous hardware.

 ## ParabolicMagneticField

The parabolic parametrisation of the magnetic field was optimised for extrapolation within R < 115 cm and |Z| < 280 cm, which is (almost) up to the surface of ECAL.
Beyond that it will diverge from the actual magnetic field map, especially at large eta, in ways that are hard to quantify.

This is based on a simple parabolic formula of Bz(z), neglecting the Br and Bphi components, and is thus vert fast. To account for the missing Br component, which contributes to transverse bending, the parametrization is tuned to reproduce as closely as possible the field integral computed over a straight path originated in (0,0,0) and across the whole tracker, rather than the value of Bz. More details can be found [here](https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideMagneticField#Alternative_parametrizations_of).



