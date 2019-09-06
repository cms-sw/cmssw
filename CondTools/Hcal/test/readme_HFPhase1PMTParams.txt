Building HFPhase1PMTParams helper executables
---------------------------------------------

The following executables assist in making and visualizing HF
dual-anode PMT timing and charge asymmetry cuts:

write_HFPhase1PMTParams    -- this program writes "HFPhase1PMTParams"
    configuration object into a boost binary file

makeHFPhase1PMTParamsPlots -- this program visualizes HF reconstruction
    cuts defined by "HFPhase1PMTParams" configuration objects

To compile these programs, set up the CMSSW environment, check out
CondTools/Hcal package, and build it.

Note that the executables will be placed in your directory
$CMSSW_BASE/test/$SCRAM_ARCH. For the time you are working with these
programs, I suggest adding this directory to your PATH environment.
In csh-like shell you can do the following (of course, assuming that
your CMSSW environment is already set up):

setenv PATH $CMSSW_BASE/test/${SCRAM_ARCH}:$PATH
rehash


Making the boost binary file
----------------------------

Run "write_HFPhase1PMTParams" executable without any command line
arguments to see its usage instructions. Then write out the filter
configuration you want into a file (name it as you like). Example:

cd $CMSSW_BASE/src/CondTools/Hcal/test
write_HFPhase1PMTParams 3 test.bbin

To change the filter configuration(s), edit the files
make_HFPhase1PMTParams_data.cc, make_HFPhase1PMTParams_mc.cc,
make_HFPhase1PMTParams_dummy.cc, and/or make_HFPhase1PMTParams_test.cc,
rebuild CondTools/Hcal package, and rerun "write_HFPhase1PMTParams".

More details about creating HFPhase1PMTParams objects are available
below, in the section "Constructing HFPhase1PMTParams".


Visualizing the cuts
--------------------

Run the program "makeHFPhase1PMTParamsPlots" without any command line
arguments to see its complete usage instructions. Example usage:

cd $CMSSW_BASE/src/CondTools/Hcal/test
makeHFPhase1PMTParamsPlots pmtlist_example.txt test.bbin pmtcuts.root

After this, you should be able to examine various plots in the
"pmtcuts.root" file with the help of "root". The titles of the plots
are self-explanatory.


Making the private database file
--------------------------------

In order to upload the HFPhase1PMTParams configuration to the CMS
database, you need to create a private mysql database file first.

Edit the file "HFPhase1PMTParamsDBWriter_cfg.py" so that the "inputfile"
variable at the beginning of that file points to the boost binary file
you created and verified in the previous steps. Also edit "database"
and "tag" variables as desired. Then run

cd $CMSSW_BASE/src/CondTools/Hcal/test
cmsRun HFPhase1PMTParamsDBWriter_cfg.py

This will create the database file (with the name specified by the
"database" variable) which will contain HFPhase1PMTParams configuration.

You should verify that the private .db file you created contains
a valid record by modifying HFPhase1PMTParamsDBReader_cfg.py
appropriately (in particular, variables "database", "tag", and
"outputfile" at the beginning of the file) and then running

cmsRun HFPhase1PMTParamsDBReader_cfg.py

The binary file written out by the database reader should be
exactly the same as the original binary file written out by
"write_HFPhase1PMTParams" (you can simply "diff" these files).


Uploading the data to the CMS database
--------------------------------------

The uploading instructions are at

https://twiki.cern.ch/twiki/bin/view/CMS/ConditionUploader

Before uploading anything, you will need to obtain appropriate
permissions, as described on that web page.

You can request a global tag for your uploaded data at

https://cms-conddb.cern.ch/cmsDbBrowser/request/Prod

After uploading your data and requesting the global tag, post a message
to the AlCa/DB hypernews: hn-cms-alca@cern.ch

Further documentation on uploading the data and requesting the global
tag can be found at

https://indico.cern.ch/event/507993/contributions/2020447/attachments/1252206/1846834/talk3_-_new_payloads__release_to_db_and_inclusion_in_global_tag.pdf


Constructing HFPhase1PMTParams
------------------------------

HFPhase1PMTParams is, basically, a lookup table which permits
an efficient lookup of "HFPhase1PMTData" objects by detector id.
You can find an example which illustrates construction of this
table in file "make_HFPhase1PMTParams_test.cc".

HFPhase1PMTParams by itself is a typedef defined in the
"CondFormats/HcalObjects/interface/HFPhase1PMTParams.h" header
file as follows:

typedef HcalItemCollById<HFPhase1PMTData> HFPhase1PMTParams;

The constructor of HFPhase1PMTParams looks, effectively, as
follows:

HFPhase1PMTParams(const HcalItemColl<HFPhase1PMTData>& coll,
                  const HcalIndexLookup& indexLookupTable,
                  const unsigned detIdTransformCode,
                  std::unique_ptr<HFPhase1PMTData> defaultItem);

This constructor is called at the very end of the
"make_HFPhase1PMTParams_test.cc" example file.

The constructor arguments are as follows:

coll               -- Effectively, a collection of HFPhase1PMTData
                      objects with a simple linear lookup of items by
                      unsigned integer index ranging from 0 to size-1.
                      It works, basically, just like std::vector. The
                      difference with std::vector is as follows:

                      1) The collection stores smart pointers instead
                         of the objects themselves. This permits
                         storage of multiple object types (all derived
                         from the same base class) in one container.
                         Pointer aliasing is allowed (i.e., multiple
                         pointers can refer to the same object).

                      2) This collection is serializable (via boost).

indexLookupTable   -- This is a lookup table from one unsigned integer
                      into another. This table is used to look up the
                      object index in "coll" by detector id. Before
                      the object index lookup is performed, the
                      detector id itself must be transformed into
                      an unsigned integer. This transformation can
                      be performed in a number of different ways.
                      The actual transformation used is specified
                      by the next argument. By functionality, this
                      table is similar to std::map<unsigned,unsigned>,
                      where the transformed detector id is the key
                      and the object index in "coll" is the value.

detIdTransformCode -- This argument specifies how to transform the
                      detector id into an unsigned integer during lookups
                      of HFPhase1PMTData objects. The available transforms
                      are listed in the file

                      CondFormats/HcalObjects/interface/HcalDetIdTransform.h

                      Possible transforms at the time of this writing are:

                      RAWID   -- Generate a unique unsigned number for
                                 every channel id.
                      
                      IETA    -- Generate a unique unsigned number for
                                 every Hcal ieta.
                      
                      IETAABS -- Generate a unique unsigned number for
                                 every Hcal |ieta|.
                      
                      SUBDET  -- Generate a unique unsigned number for
                                 every Hcal subdetector.

                      You will want to use the transform which corresponds
                      to the granularity with which you would like to
                      define the HFPhase1PMTData cuts. For example, if you
                      want to define the same cuts for all PMTs corresponding
                      to the same ieta, use the IETA transform. If, at the
                      same time, the constants are the same for both HF+
                      and HF-, use the IETAABS transform. If the cuts are
                      different for all PMTs, use the RAWID transform, etc.
                      Use of a proper transform minimizes the number of
                      entries that have to be stored in the "indexLookupTable"
                      and speeds up the lookup. You can also add your own
                      transforms -- this will not invalidate existing
                      database records. See the comments in the
                      HcalDetIdTransform.h header file for details.

defaultItem        -- This item will be used in case "indexLookupTable"
                      does not contain a key corresponding to the given
                      detector id. If you think that your "indexLookupTable"
                      argument provides proper lookups for all PMTs,
                      simply give an empty pointer here. In this case there
                      will be no default, and the code will crash in case
                      you made a mistake in constructing "indexLookupTable".

Internally, when the code is using HFPhase1PMTParams and fetching
HFPhase1PMTData by id, the lookup is performed as follows:

1) The id is transformed according to detIdTransformCode. If you
   are curious how, exactly, this is done, read the code in
   CondFormats/HcalObjects/src/HcalDetIdTransform.cc. Let say,
   the result of this operation is unsigned int "tid".

2) The indexLookupTable is invoked to look up the linear index inside
   "coll" by "tid". Let say, the result of this lookup is "ind".
   Note that this lookup is allowed to fail (there may be no value
   corresponding to "tid" in the collection).

3) If the lookup during the previous step did not fail, the item with
   index "ind" is extracted from "coll" and returned. If the lookup
   during the previous step did fail, the default item is returned.
   If the lookup during the previous step did fail and there is no
   default, the code will throw an exception.

To create "coll", make an object of type HcalItemColl<HFPhase1PMTData>
using the default constructor and then "push_back" items of type
std::unique_ptr<HFPhase1PMTData> (the collection will assume ownership
of these items).

To create "indexLookupTable", make an object of type "HcalIndexLookup"
using the default constructor, and then call the "add" method of the
object for each key-value pair you want to insert. Note that the keys
must be unique, and the values must be less than the size of "coll".
In order to make sure that the keys are correct, generate them with
the "HcalDetIdTransform::transform" function utilizing the transform
code with which HFPhase1PMTParams will be subsequently created.


Constructing HFPhase1PMTData objects
------------------------------------

An item of type HFPhase1PMTData specifies the PMT cuts to be applied
during HF local reconstruction. These are the TDC timing cuts (applied
to each anode) and the charge asymmetry cut (applied to the PMT as
a whole). HFPhase1PMTData constructor looks as follows:

HFPhase1PMTData(const Cuts& cutShapes, const float charge0,
                const float charge1, const float minQAsymm);

The constructor arguments are:

cutShapes -- An array of cut shapes. "Cuts" is actually a typedef:
             typedef boost::array<std::shared_ptr<AbsHcalFunctor>,6> Cuts.
             AbsHcalFunctor is a base class for functors which define
             an arbitrary univariate function. In this case, the functors
             describe the dependence of the cut value on the collected
             charge. The cuts are identified by their indices in the array.
             These indices are described in detail in the header file
             CondFormats/HcalObjects/interface/HFPhase1PMTData.h.
             There are two timing cuts (min and max) for each anode and
             two asymmetry cuts (also min and max).

charge0   -- Minimum charge that has to be collected by the first anode
             (mapped to depth 1 and 2) for a reliable time measurement.
             If the charge is less than this value, the TDC timing cuts
             will not be applied, and the anode will not participate in
             the determination of the rechit time.

charge1   -- Minimum charge that has to be collected by the second anode
             (mapped to depth 3 and 4) for a reliable time measurement.

minQAsymm -- Minimum PMT combined charge needed in order to apply the
             charge asymmetry cut. If the charge is less than this value,
             the asymmetry cut will not be applied.

Available cut shape classes (derived from AbsHcalFunctor) are:

HcalConstFunctor             -- This class can be used to represent cuts
                                which do not depend on charge.

HcalChebyshevFunctor         -- This class implements Chebyshev polynomial
                                series on some interval [Qmin, Qmax].
                                Outside this interval the cut is assumed
                                to be constant.

HcalCubicInterpolator        -- This class implements cubic Hermite spline
                                (both the function values and the derivatives
                                are known at a set of points). See
                             http://en.wikipedia.org/wiki/Cubic_Hermite_spline

HcalInterpolatedTableFunctor -- Cuts are represented by a piecewise linear
                                function. The function values are provided
                                for a set of equidistant points.

HcalLinearCompositionFunctor -- A functor returning a linearly transformed
                                value of another functor: f(Q) = a*p(Q) + b.
                                Useful for implementing cuts symmetric
                                about 0, etc. Note that, due to a bug in
                                the boost library employed by CMSSW, this
                                class can not be used at the time of this
                                writing (Sep 2016). Please wait until CMSSW
                                boost is updated to version 1.59 or newer.

HcalPiecewiseLinearFunctor   -- Cuts are represented by a piecewise linear
                                function. The function values can be provided
                                for a set of N arbitrarily spaced points.
                                Naturally, calculation of the function values
                                by this class is slower than by the class
                                HcalInterpolatedTableFunctor (O(log(N))
                                computational complexity instead of O(1)).

HcalPolynomialFunctor        -- Cuts are represented by polynomial series
                                in the monomial basis for the variable
                                (Q + Qshift), with some arbitrary
                                configurable Qshift.

The header files for all of these classes can be found in the directory
CondFormats/HcalObjects/interface.


Making a standalone package
---------------------------

Run the script "extract_HFPhase1PMTParams.tcl" in order to create
a standalone software package containing the code for creating
HFPhase1PMTParams boost binary files and for cut visualization.
This package will be archived in the file "HFPhase1PMTParams.tar.gz".
The standalone code requires boost and root for compilation but not
CMSSW. The build procedure in the package (Makefile) is known to work
for Ubuntu Linux. On your computer, you might need to adjust variables
BOOST_INC, BOOST_LIB, and LIBS in files "Makefile" and "Makefile.plots".
(e.g., change /usr/lib into /usr/lib64, etc). Run

make

in order to build the "write_HFPhase1PMTParams" executable and

make -f Makefile.plots clean
make -f Makefile.plots

in order to build "makeHFPhase1PMTParamsPlots". It should be easy to
combine this package with an analysis code that generates TDC time
and charge asymmetry cuts.

If you use the standalone code to generate HFPhase1PMTParams, make
sure that the version of "boost" library on your computer is not newer
than the version of "boost" used by CMSSW. Otherwise you will not be
able to create the private mysql database for uploading the cuts.
