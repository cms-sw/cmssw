#!/usr/bin/perl -w
use strict;
# make sata mapping of volume serial numbers, suitable for intput to "makeall" script

#define list of Satabeasts (order implicitly matched to PC's in @nodeAssign):
my @satabeasts = (
		  "SATAB-C2C07-03-00",
		  "SATAB-C2C07-04-00",
		  "SATAB-C2C07-05-00");

#nominal node mapping to satabeasts (ordering tied to that in @satabeasts):
my @nodeAssign = (
		  "srv-C2C07-19",
		  "srv-C2C07-20",
		  "srv-C2C07-13",
		  "srv-C2C07-14",
		  "srv-C2C07-15",
		  "srv-C2C07-16");


my $nbeast=0;
my $VolPerSata;
my @VolumeID;
my @VolumeSerial;
foreach my $ibeast ( @satabeasts ) {
    
# Get Vol Name:
    my $command = " lynx -connect_timeout=2 -dump -auth=USER:mickey2mouse 'http://$ibeast/hluninf.asp?detail&rpath=/vwlunmap.asp' | grep 'Volume name'";
    my @volIDs = `$command`;
    
    $VolPerSata=0;
foreach my $volline ( @volIDs) {    
    chomp($volline);    
    my @split = split(m/name/, $volline);
    $VolumeID[$VolPerSata] = $split[1];
    $VolPerSata++;
}
    
#Get Vol Serial Number:
 $command = " lynx -connect_timeout=2 -dump -auth=USER:mickey2mouse 'http://$ibeast/hluninf.asp?detail&rpath=/vwlunmap.asp'  | grep 'Volume serial number'";
    my @volSerl = `$command`;
    
    my $SerialPerSata=0;
    foreach my $volline ( @volSerl) {
	chomp($volline);    
	my @split = split(m/number/, $volline);
	$VolumeSerial[$SerialPerSata] =  $split[1];
	$VolumeSerial[$SerialPerSata] =~ tr/A-Z/a-z/;
	$SerialPerSata++;
    }
    
# if there is a mismatch, go into error:
    if($VolPerSata != $SerialPerSata){
	print "\n !!!!! Something is VERY WRONG: Number of Vol Names NOT Equal to Number of ID's !!!!!! \n";
	print "\n !!!!!          $VolPerSata Vols  vs  $SerialPerSata Serial No.        !!!!!! \n\n";
	exit;
    }
    
    
    
    $nbeast++;
    my $nvol = $VolPerSata;
    
    
#------------dump everything together:
    for( my $i=0; $i<$VolPerSata; $i++){
	
# compute index to point to right PC as owner of Satabeast
# ASSUME half of all volumes go to each PC
	my $mod;   {use integer;   $mod = (2*$i)/$VolPerSata; }
	my $pcIndx = 2*($nbeast-1)+$mod;
#Switch to lower case:    
	$VolumeID[$i] =~ s/([a-zA-Z]*)([0-9][0-9])([a-zA-Z]+)([0-9]+)(.*)/$1$2 $3$4 $5/;
	print "$VolumeID[$i]  $VolumeSerial[$i]  $nodeAssign[$pcIndx] \n"; 
    }
    

}

exit;

