#!/usr/bin/env perl
# $Id: mk_satamap.pl,v 1.7 2009/08/17 13:26:47 gbauer Exp $
#
# Make sata mapping of volume serial numbers, suitable for intput to "makeall" script
#

use strict;

#define list of Satabeasts (order implicitly matched to PC's in @nodeAssign):
my @satabeasts = (
                  "SATAB-C2C06-03-00",
                  "SATAB-C2C06-04-00",
                  "SATAB-C2C06-05-00",
                  "SATAB-C2C06-06-00",
#
		  "SATAB-C2C07-03-00",
		  "SATAB-C2C07-04-00",
		  "SATAB-C2C07-05-00",
		  "SATAB-C2C07-06-00",
#
		  "SATAB-C2f37-01-00"
                  );

#nominal node mapping to satabeasts (ordering tied to that in @satabeasts):
my @nodeAssign = (
                  "srv-C2C06-12",
                  "srv-C2C06-13",
                  "srv-C2C06-14",
                  "srv-C2C06-15",
                  "srv-C2C06-16",
                  "srv-C2C06-17",
                  "srv-C2C06-18",
                  "srv-C2C06-19",
#
		  "srv-C2C07-12",
		  "srv-C2C07-13",
		  "srv-C2C07-14",
		  "srv-C2C07-15",
		  "srv-C2C07-16",
		  "srv-C2C07-17",
		  "srv-C2C07-18",
		  "srv-C2C07-19",
#
		  "dvsrv-C2f37-01",
		  "dvsrv-C2f37-02"
                  );

# wwpn for each node (can be obtained with getwwpn.sh)
my %nodeWWPN = ( 
                 "srv-C2C06-12" => "21-00-00-1B-32-1B-AB-87      21-01-00-1B-32-3B-AB-87",
                 "srv-C2C06-13" => "21-00-00-1B-32-1B-19-C9      21-01-00-1B-32-3B-19-C9",
                 "srv-C2C06-14" => "21-00-00-1B-32-00-5B-D2      21-01-00-1B-32-20-5B-D2",
                 "srv-C2C06-15" => "21-00-00-1B-32-1B-A3-85      21-01-00-1B-32-3B-A3-85",
                 "srv-C2C06-16" => "21-00-00-1B-32-0F-22-E6      21-01-00-1B-32-2F-22-E6",
                 "srv-C2C06-17" => "21-00-00-1B-32-1B-C5-87      21-01-00-1B-32-3B-C5-87",
                 "srv-C2C06-18" => "21-00-00-1B-32-1B-3E-8B      21-01-00-1B-32-3B-3E-8B",
                 "srv-C2C06-19" => "21-00-00-1B-32-1B-E6-85      21-01-00-1B-32-3B-E6-85",
#
		 "srv-C2C07-12" => "21-00-00-1B-32-13-BF-B4      21-01-00-1B-32-33-BF-B4",
		 "srv-C2C07-13" => "21-00-00-1B-32-1B-36-85      21-01-00-1B-32-3B-36-85",
		 "srv-C2C07-14" => "21-00-00-1B-32-13-77-CC      21-01-00-1B-32-33-77-CC",
		 "srv-C2C07-15" => "21-00-00-1B-32-0F-40-F5      21-01-00-1B-32-2F-40-F5",
		 "srv-C2C07-16" => "21-00-00-1B-32-13-FD-B0      21-01-00-1B-32-33-FD-B0",
		 "srv-C2C07-17" => "21-00-00-1B-32-13-3E-C5      21-01-00-1B-32-33-3E-C5",
		 "srv-C2C07-18" => "21-00-00-1B-32-1B-4B-87      21-01-00-1B-32-3B-4B-87",
		 "srv-C2C07-19" => "21-00-00-1B-32-13-D1-CA      21-01-00-1B-32-33-D1-CA",
		 "srv-C2C07-20" => "21-00-00-1B-32-1B-50-3B      21-01-00-1B-32-3B-50-3B",
#spare PCs:
                 "srv-C2C06-20" => "21-00-00-1B-32-1B-58-86      21-01-00-1B-32-3B-58-86",
		 "srv-C2C07-20" => "21-00-00-1B-32-1B-50-3B      21-01-00-1B-32-3B-50-3B",
#DaqVal
                 "dvsrv-C2f37-01" => "21-00-00-E0-8B-9D-37-F1      21-00-00-E0-8B-9D-37-F1",
                 "dvsrv-C2f37-02" => "21-00-00-E0-8B-9E-68-0C      21-00-00-E0-8B-9E-68-0C"
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
