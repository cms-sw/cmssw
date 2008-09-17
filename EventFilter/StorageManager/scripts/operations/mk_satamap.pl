#!/usr/bin/perl -w
# $Id: mk_satamap.pl,v 1.3 2008/09/13 01:06:45 loizides Exp $
#
# Make sata mapping of volume serial numbers, suitable for intput to "makeall" script
#

use strict;

#define list of Satabeasts (order implicitly matched to PC's in @nodeAssign):
my @satabeasts = (
		  "SATAB-C2C07-03-00",
		  "SATAB-C2C07-04-00",
		  "SATAB-C2C07-05-00"
                  );

#nominal node mapping to satabeasts (ordering tied to that in @satabeasts):
my @nodeAssign = (
		  "srv-C2C07-19",
		  "srv-C2C07-20",
		  "srv-C2C07-13",
		  "srv-C2C07-14",
		  "srv-C2C07-18",
		  "srv-C2C07-16"
                  );

# wwpn for each node (can be obtained with getwwpn.sh)
my %nodeWWPN = ( 
                 "srv-C2C07-13" => "21-00-00-E0-8B-9D-A9-F4",
                 "srv-C2C07-14" => "21-00-00-E0-8B-9D-37-F1",
                 "srv-C2C07-15" => "unknown",
                 "srv-C2C07-16" => "21-00-00-E0-8B-9E-68-0C",
                 "srv-C2C07-17" => "21-00-00-E0-8B-9D-BA-F3",
                 "srv-C2C07-18" => "21-00-00-E0-8B-9D-C3-F3",
                 "srv-C2C07-19" => "21-00-00-E0-8B-9D-A5-F1",
                 "srv-C2C07-20" => "21-00-00-E0-8B-9D-0C-F2"
                 );

my $nbeast=0;
my $VolPerSata;
my @VolumeID;
my @VolumeSerial;

foreach my $ibeast ( @satabeasts ) {
    
    # get volume name
    my $command = " lynx -connect_timeout=2 -dump -auth=USER:mickey2mouse 'http://$ibeast/hluninf.asp?detail&rpath=/vwlunmap.asp' | grep 'Volume name'";
    my @volIDs = `$command`;
    
    $VolPerSata=0;
    foreach my $volline ( @volIDs) {    
        chomp($volline);    
        my @split = split(m/name/, $volline);
        $VolumeID[$VolPerSata] = $split[1];
        $VolPerSata++;
    }
    
    #get vol serial number
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
        my $nodename = "$nodeAssign[$pcIndx]";
        # switch to lower case:    
	$VolumeID[$i] =~ s/([a-zA-Z]*)([0-9][0-9])([a-zA-Z]+)([0-9]+)(.*)/$1$2 $3$4 $5/;
	print "$VolumeID[$i]  $VolumeSerial[$i]  $nodename  $nodeWWPN{$nodename}\n"; 
    }
}

exit;
