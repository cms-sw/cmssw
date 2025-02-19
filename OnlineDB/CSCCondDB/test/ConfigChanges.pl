#!/usr/bin/env perl

#Get the RUI name
$RUI_DUMMY = substr($ARGV[0],30);
$RUI = substr($RUI_DUMMY,0, 5);

#This is what we begin with
@DSOURCE_ARRAY=(
"untracked vstring RUI00 = {}",
"untracked vstring RUI01 = {}",
"untracked vstring RUI02 = {}",
"untracked vstring RUI03 = {}",
"untracked vstring RUI04 = {}",
"untracked vstring RUI05 = {}",
"untracked vstring RUI06 = {}",
"untracked vstring RUI07 = {}",
"untracked vstring RUI08 = {}",
"untracked vstring RUI09 = {}",
"untracked vstring FED750 = {}",
"untracked vstring FED751 = {}",
"untracked vstring FED752 = {}",
"untracked vstring FED753 = {}",
"untracked vstring FED754 = {}",
"untracked vstring FED755 = {}",
"untracked vstring FED756 = {}",
"untracked vstring FED757 = {}",
"untracked vstring FED758 = {}",
"untracked vstring FED759 = {}",
"untracked vstring FED760 = {}"
);

%MAP=("RUI00",FED750,
      "RUI01",FED751,
      "RUI02",FED752,
      "RUI03",FED753,
      "RUI04",FED754,
      "RUI05",FED755,
      "RUI06",FED756,
      "RUI07",FED757,
      "RUI08",FED758,
      "RUI09",FED759);

#loop over the full array from above
#put all lines except the one with the RUI into the input variable
$MAP_VAR = $MAP{"$RUI"};
foreach (@DSOURCE_ARRAY){
 if( ($_ !~ $RUI ) && ($_ !~ $MAP_VAR) ) {
     $DSOURCE_INPUT = $DSOURCE_INPUT . $_  . "\n";
 };
 #with the others...
 if ($_ =~ $RUI){
     $DSOURCE_INPUT = $DSOURCE_INPUT . "untracked vstring " . $MAP_VAR . " ={\'$RUI\'}" . "\n";
 };
};

open(CONFIGUPDATE, ">ConfigUpdate.txt");
print CONFIGUPDATE "$DSOURCE_INPUT";
close(CONFIGUPDATE); 

