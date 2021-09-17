# RecoLuminosity/LumiProducer

This package contains utilities for producing luminosity information in CMSSW. Most of the plugins here are obsolete and only for Run 1; they are kept here for backwards compatibility but are not actively maintained.

For Run 2, the plugin `LumiProducerFromBrilcalc` is available. This allows you to add luminosity information to your CMSSW job by reading in a CSV output file produced by the `brilcalc` utility. In order to use it, add the following to your `cfg` file:

```
process.LumiInfo = cms.EDProducer('LumiProducerFromBrilcalc',
                                  lumiFile = cms.string("./myLumiFile.csv"),
                                  throwIfNotFound = cms.bool(False),
                                  doBunchByBunch = cms.bool(False))
```
where `lumiFile` is the output file created by `brilcalc`, `throwIfNotFound` will determine the behavior if the csv file does not contain information for an event in your input file (if `True`, an exception will be thrown; if `False`, the luminosity will just be taken to be zero for that event), and `doBunchByBunch` should remain `False`, since bunch-by-bunch luminosity is not currently supported.

Add `LumiInfo` to your path, and then you can access the `LumiInfo` object produced under the input tag `("LumiInfo", "brilcalc")`.

For more information on the proper way to use `brilcalc` to produce a luminosity csv file, please consult the [LumiPOG twiki](https://twiki.cern.ch/twiki/bin/view/CMS/TWikiLUM#LumiCMSSW).

												      