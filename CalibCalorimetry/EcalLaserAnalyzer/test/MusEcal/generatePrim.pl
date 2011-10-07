#!/usr/bin/env perl

use Term::ANSIColor;
#use Date::Manip; 
use Cwd;

$cfgFile     = @ARGV[0];
$ecalPart    = @ARGV[1];

print " config: $cfgFile \n";
print " ecalPart: $ecalPart \n";

do "/nfshome0/ecallaser/config/readconfig.pl";
readconfig(${cfgFile});

${MON_MUSECAL_DIR}=~ s/\s+//;
${MON_OUTPUT_DIR}=~ s/\s+//;
${LMF_LASER_PERIOD}=~ s/\s+//;

print "MON_MUSECAL_DIR  : ${MON_MUSECAL_DIR} \n";
print "MON_OUTPUT_DIR: ${MON_OUTPUT_DIR} \n";
print "LMF_LASER_PERIOD: ${LMF_LASER_PERIOD} \n";

while( 1 ) 
{   
    
    generate();
    sleep 10;
}


sub generate{
    
    my $command1 = "${MON_MUSECAL_DIR}/generatePrim.sh ${cfgFile} ${ecalPart} >> ${MON_OUTPUT_DIR}/${LMF_LASER_PERIOD}/log/generatePrimSh$ecalPart.log";
    system $command1;	
    
#    my $command2 = "${MON_MUSECAL_DIR}/generateCLS.sh ${cfgFile} >> ${MON_OUTPUT_DIR}/${LMF_LASER_PERIOD}/log/generateCLSSh.log";
#    system $command2;	
}

exit;

