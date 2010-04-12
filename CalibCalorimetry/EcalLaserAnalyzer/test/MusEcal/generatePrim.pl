#!/usr/bin/env perl

use Term::ANSIColor;
#use Date::Manip; 
use Cwd;

$cfgFile     = @ARGV[0];

do "/nfshome0/ecallaser/config/readconfig.pl";
readconfig(${cfgFile});

${MON_MUSECAL_DIR}=~ s/\s+//;
${MON_OUTPUT_DIR}=~ s/\s+//;
${LMF_LASER_PERIOD}=~ s/\s+//;
while( 1 ) 
{   
    
    generate();
    sleep 60;
}


sub generate{
    
    my $command1 = "${MON_MUSECAL_DIR}/generatePrim.sh ${cfgFile} > ${MON_OUTPUT_DIR}/${LMF_LASER_PERIOD}/log/generatePrimSh.log";
    system $command1;	
    
}

exit;

