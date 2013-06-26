#!/usr/bin/env perl

use Term::ANSIColor;
#use Date::Manip; 
use Cwd;

$linkName     = @ARGV[0];

#./Cosmics09_310/musecal/generatePrim.sh Cosmics09_310

while( 1 ) 
{   
    
    generate();
    sleep 60;
}


sub generate{
    
    my $command1 = "./${linkName}/musecal/generatePrim.sh ${linkName} > ${linkName}/log/generatePrimSh.log";
    system $command1;	
    
}

exit;

