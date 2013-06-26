#! /bin/sh

echo "******* Checking Severity ERROR"
egrep "\%MSG-.*"   ./$1/**/*.log | \
gawk '{print $1 $2}' | \
perl -ne 'BEGIN{%sys=()} if(m#.\/(\d+)\/*(\d+)\/(.*?:)\%MSG-[e](.*):#) {push @{$sys{$2}},$4;} END{foreach $test (keys %sys) {print "$test\t". scalar @{$sys{$test}} . " @{$sys{$test}}\n" }}' | \
sort -n 

echo "******* Checking Severity WARNING"
egrep "\%MSG-.*"   ./$1/**/*.log | \
gawk '{print $1 $2}' | \
perl -ne 'BEGIN{%sys=()} if(m#.\/(\d+)\/*(\d+)\/(.*?:)\%MSG-[w](.*):#) {push @{$sys{$2}},$4;} END{foreach $test (keys %sys) {print "$test\t". scalar @{$sys{$test}} . " @{$sys{$test}}\n" }}' | \
sort -n 
