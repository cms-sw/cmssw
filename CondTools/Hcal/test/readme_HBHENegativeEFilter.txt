Building write_HBHENegativeEFilter helper executable:
-----------------------------------------------------

The program "write_HBHENegativeEFilter" writes "HBHENegativeEFilter"
configuration object into a boost binary (or text) archive.

To compile this program, set up the CMSSW environment, check out
CondTools/Hcal package, and build it.

Note that the executable will be placed in your directory
$CMSSW_BASE/test/$SCRAM_ARCH. For the time you are working with this
program, I suggest adding this directory to your PATH environment.
In csh-like shell you can do the following (of course, assuming that
your CMSSW environment is already set up):

setenv PATH $CMSSW_BASE/test/${SCRAM_ARCH}:$PATH
rehash


Making the boost binary file
----------------------------

Run "write_HBHENegativeEFilter" executable without any command line
arguments to see its usage instructions. Then write out the filter
configuration you want into a file (name it as you like). Example:

cd $CMSSW_BASE/src/CondTools/Hcal/test
write_HBHENegativeEFilter 0 test.bbin

To change the filter configuration, edit the file make_HBHENegativeEFilter.cc,
rebuild CondTools/Hcal package, and rerun "write_HBHENegativeEFilter".


Making the private database file
--------------------------------

In order to upload the HBHENegativeEFilter configuration to the CMS
database, you need to create a private mysql database file first.

Edit the file "HBHENegativeEFilterDBWriter_cfg.py" so that the "inputfile"
variable at the beginning of that file points to the boost binary file
you created and verified in the previous steps. Also edit "database"
and "tag" variables as desired. Then run

cd $CMSSW_BASE/src/CondTools/Hcal/test
cmsRun HBHENegativeEFilterDBWriter_cfg.py

This will create the database file (with the name specified by the
"database" variable) which will contain HBHENegativeEFilter configuration.

You should verify that the private .db file you created contains
a valid record by modifying HBHENegativeEFilterDBReader_cfg.py
appropriately (in particular, variables "database", "tag", and
"outputfile" at the beginning of the file) and then running

cmsRun HBHENegativeEFilterDBReader_cfg.py

The binary file written out by the database reader should be
exactly the same as the original binary file written out by
"write_HBHENegativeEFilter" (you can simply "diff" these files).


Uploading the data to the CMS database
--------------------------------------

The instructions can be found in file "readme_HFPhase1PMTParams.txt",
inside identically named section.


Making a standalone package
---------------------------

Run the script "extract_HBHENegativeEFilter.tcl" in order to create
a standalone software package containing the code for creating
HBHENegativeEFilter boost binary files. This package will be archived
in the file "HBHENegativeEFilter.tar.gz". The standalone code requires
boost for compilation but not CMSSW. The build procedure in the
package (Makefile) is known to work on Ubuntu Linux. On your computer,
you might need to adjust variables BOOST_INC, BOOST_LIB, and LIBS in
the Makefile (e.g., change /usr/lib into /usr/lib64, etc). Run "make"
in order to build the "write_HBHENegativeEFilter" executable.

If you use the standalone code to generate HBHENegativeEFilter, make
sure that the version of "boost" library on your computer is not newer
than the version of "boost" used by CMSSW. Otherwise you will not be
able to create the private mysql database for uploading the negative
energy discriminant.
