import FWCore.ParameterSet.VarParsing as VarParsing

options = VarParsing.VarParsing()
options.register('runNumber',
                1,
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.int,
                "the run number to be uploaded.")
options.register('destinationDatabase',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "the destination database connection string.")
options.register('destinationTag',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "the destination tag name.")
options.register('tagForRunInfo',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "the tag name used to retrieve the RunInfo payload and the magnet current therein.")
options.register('tagForBOff',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "the tag name used to retrieve the reference payload for magnet off.")
options.register('tagForBOn',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "the tag name used to retrieve the reference payload for magnet on.")
options.register('currentThreshold',
                7000.0,
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.float,
                "the threshold on the magnet current for considering a switch of the magnetic field.")
options.parseArguments()
