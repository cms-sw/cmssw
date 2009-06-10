#!/usr/bin/env perl
use strict;

die "Usage $0 host port class instance paramName paramType paramValue\n" if @ARGV != 7;

my $host       = $ARGV[0];
my $port       = $ARGV[1];
my $class      = $ARGV[2];
my $instance   = $ARGV[3];
my $paramName  = $ARGV[4];
my $paramType  = $ARGV[5];
my $paramValue = $ARGV[6];


my $m = '<soap-env:Envelope soap-env:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/" xmlns:soap-env="http://schemas.xmlsoap.org/soap/envelope/" xmlns:soapenc="http://schemas.xmlsoap.org/soap/encoding/" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">';
$m .= '<soap-env:Body>';
$m .= '<xdaq:ParameterSet xmlns:xdaq="urn:xdaq-soap:3.0">';
$m .= '<p:properties xmlns:p="urn:xdaq-application:CLASS" xsi:type="soapenc:Struct">';
$m .= '<p:PARAM_NAME xsi:type="xsd:PARAM_TYPE">PARAM_VALUE</p:PARAM_NAME>';
$m .= '</p:properties>';
$m .= '</xdaq:ParameterSet>';
$m .= '</soap-env:Body>';
$m .= '</soap-env:Envelope>';


$m =~ s/CLASS/$class/g;
$m =~ s/PARAM_NAME/$paramName/g;
$m =~ s/PARAM_TYPE/$paramType/g;
$m =~ s/PARAM_VALUE/$paramValue/g;

$m =~ s/"/\\\"/g;

print "MESSAGE=$m\n";

my $curlCmd  = "curl --stderr /dev/null -H \"Content-Type: text/xml\" -H \"Content-Description: SOAP Message\" -H \"SOAPAction: urn:xdaq-application:class=$class,instance=$instance\" http://$host:$port -d \"$m\"";

open CURL, "$curlCmd|";

my $reply = "";

while(<CURL>) {
  chomp;
  $reply .= $_;
}

#print "\n\n\nreply\n\n$reply\n\n";

if($reply =~ m#<\w+:ParameterSetResponse\s#) {
  print "$reply\n";
  exit 0;
} else {
  print "ERROR\n\n$reply\n\n";
  exit 1;
}  

 
exit;

