#!/usr/bin/env perl
use DirHandle;
#print $#ARGV;
($#ARGV >= 2) || die "Usage::install_env.pl libdir moddir conffile";
$libdir = @ARGV[0];
$moddir = @ARGV[1];
$confpath = @ARGV[2];
!(-e $libdir) || die "ERROR::$libdir exists. Please remove it or choose another name"; 
!(-e $moddir) || die "ERROR::$moddir exists. Please remove it or choose another name"; 
$pid = $$;
$envfile = "/tmp/cmsswenv$pid.tmp";
print "creating emvironment file $envfile\n";
system `scramv1 runtime -sh > $envfile`;
# fix for scram output in 2_1_X ?
system `echo "export SEAL_PLUGINS=\"$ENV{SEAL_PLUGINS}\"\;" >> $envfile`;
system `echo "export CMSSW_RELEASE_BASE=\"$ENV{CMSSW_RELEASE_BASE}\"\;" >> $envfile`;
system `echo "export ORACLE_HOME=\"$ENV{ORACLE_HOME}\"\;" >> $envfile`;
system `echo "export TNS_ADMIN=\"$ENV{TNS_ADMIN}\"\;" >> $envfile`;
system `echo "export ROOTSYS=\"$ENV{ROOTSYS}\"\;" >> $envfile`;
system `echo "export SEAL=\"$ENV{SEAL}\"\;" >> $envfile`;
system `echo "export SEAL_KEEP_MODULES=\"$ENV{SEAL_KEEP_MODULES}\"\;" >> $envfile`;
system `echo "export POOL_OUTMSG_LEVEL=\"$ENV{POOL_OUTMSG_LEVEL}\"\;" >> $envfile`;
system `echo "export POOL_STORAGESVC_DB_AGE_LIMIT=\"$ENV{POOL_STORAGESVC_DB_AGE_LIMIT}\"\;" >> $envfile`;
system `echo "export CMSSW_DATA_PATH=\"$ENV{CMSSW_DATA_PATH}\"\;" >> $envfile`;
system `echo "export CMSSW_SEARCH_PATH=\"$ENV{CMSSW_SEARCH_PATH}\"\;" >> $envfile`;

print "creating library clones at $libdir\n";
`copyFUlibs.pl $envfile $libdir`;
print "creating module clones at $moddir\n";
`copyFUmodules.pl $envfile $moddir`;
print "fixing libs and mods at $libdir, $moddir\n";
`fix_libs.pl $libdir $moddir`;
print "recreating edm cache file in $libdir\n";
`reset_edm_cache.pl $libdir $moddir --`;
print "recreating seal cache file in $moddir\n";
`reset_seal_cache.pl $libdir $moddir --`;
print "preparing config string for DuckCad at $confpath\n";
open(CONF,"$envfile");
open(OUT, ">$confpath");
print OUT "LD_LIBRARY_PATH=$libdir ";
print OUT "SEAL_PLUGINS=$moddir ";
while(<CONF>) {
    s/export (.+)/$1/;
    s/;//;
    s/"//g;
    s/\n//;
    print OUT "$_ " if /XDAQ_ROOT/;
    print OUT "$_ " if !/^SCRAMRT/ & /POOL_STORAGESVC_DB_AGE_LIMIT/;
    print OUT "$_ " if !/^SCRAMRT/ & /SEAL=/;
    print OUT "$_ " if !/^SCRAMRT/ & /SEAL_KEEP_MODULES/;
    print OUT "$_ " if !/^SCRAMRT/ & /ROOTSYS/;
    print OUT "$_ " if !/^SCRAMRT/ & /POOL_OUTMSG_LEVEL/;
    print OUT "$_ " if !/^SCRAMRT/ & /CMSSW_VERSION/;
    print OUT "$_ " if !/^SCRAMRT/ & /CMSSW_BASE/;
    print OUT "$_ " if !/^SCRAMRT/ & /TNS_ADMIN/;
    print OUT "$_ " if !/^SCRAMRT/ & /CMSSW_SEARCH_PATH/;
}
