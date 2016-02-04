#!/usr/bin/env perl
use strict;

die "Usage $0 host port class instance paramName paramType\n" if (@ARGV != 6 && @ARGV != 4);

my $host      = $ARGV[0];
my $port      = $ARGV[1];
my $class     = $ARGV[2];
my $instance  = $ARGV[3];
if ( @ARGV == 4 ) {
    my $params = parameterQuery( $host, $port,$class,$instance );
    my $ix = 1;
    print "\n";
    foreach my $para (@$params) {
	printf( "%3d) %25s = %-30s (%s)\n", $ix++,$para->{'name'},$para->{'value'},$para->{'type'} );
    }
    exit;
}


my $paramName = $ARGV[4];
my $paramType = $ARGV[5];


my $m  = '<SOAP-ENV:Envelope';
$m .= ' SOAP-ENV:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/"';
$m .= ' xmlns:SOAP-ENV="http://schemas.xmlsoap.org/soap/envelope/"';
$m .= ' xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"';
$m .= ' xmlns:xsd="http://www.w3.org/2001/XMLSchema"';
$m .= ' xmlns:SOAP-ENC="http://schemas.xmlsoap.org/soap/encoding/">';
$m .=   '<SOAP-ENV:Body>';
$m .=     '<xdaq:ParameterGet xmlns:xdaq="urn:xdaq-soap:3.0">';
$m .=       '<p:properties';
$m .=         ' xmlns:p="urn:xdaq-application:CLASS"';
$m .=         ' xsi:type="soapenc:Struct">';
$m .=         '<PARAM_NAME xsi:type="xsd:PARAM_TYPE">';
$m .=            'PARAM_NAME';
$m .=         '</PARAM_NAME>';
$m .=        '</p:properties>';
$m .=     '</xdaq:ParameterGet>';
$m .=   '</SOAP-ENV:Body>';
$m .= '</SOAP-ENV:Envelope>';

$m =~ s/CLASS/$class/g;
$m =~ s/PARAM_NAME/$paramName/g;
$m =~ s/PARAM_TYPE/$paramType/g;

$m =~ s/"/\\\"/g;

# print "MESSAGE=$m\n";

my $curlCmd  = "curl --stderr /dev/null -H \"Content-Type: text/xml\" -H \"Content-Description: SOAP Message\" -H \"SOAPAction: urn:xdaq-application:class=$class,instance=$instance\" http://$host:$port -d \"$m\"";

open CURL, "$curlCmd|";

my $reply = "";

while(<CURL>) {
  chomp;
  $reply .= $_;
}

if($reply =~ m#<\w+:$paramName\s[^>]*>([^<]*)</\w+:$paramName>#) {
  my $paramValue = $1;
  print "$paramValue\n";
  exit 0;
} elsif($reply =~ m#<\w+:$paramName\s[^>]*/>#) {
  print "\n";
} else {
  print "ERROR";
  print " COMMAND=";
  print "$m\n";
  print " REPLY=";
  if($reply eq "") {
    print "NONE\n";
  } else {
    print "$reply\n";
  }
  exit 1;
}
 
exit;

sub parameterQuery {
    my ($host,$port,$class,$instance ) = @_;

    my $m  = '<SOAP-ENV:Envelope';
    $m .= ' SOAP-ENV:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/"';
    $m .= ' xmlns:SOAP-ENV="http://schemas.xmlsoap.org/soap/envelope/"';
    $m .= ' xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"';
    $m .= ' xmlns:xsd="http://www.w3.org/2001/XMLSchema"';
    $m .= ' xmlns:SOAP-ENC="http://schemas.xmlsoap.org/soap/encoding/">';
    $m .=   '<SOAP-ENV:Body>';
    $m .=     '<xdaq:ParameterQuery xmlns:xdaq="urn:xdaq-soap:3.0">';
    $m .=     '</xdaq:ParameterQuery>';
    $m .=   '</SOAP-ENV:Body>';
    $m .= '</SOAP-ENV:Envelope>';
    
    $m =~ s/CLASS/$class/g;
    
    $m =~ s/"/\\\"/g;

    # print "MESSAGE=$m\n";

    my $curlCmd  = "curl --stderr /dev/null -H \"Content-Type: text/xml\" -H \"Content-Description: SOAP Message\" -H \"SOAPAction: urn:xdaq-application:class=$class,instance=$instance\" http://$host:$port -d \"$m\"";

    open CURL, "$curlCmd|";

    while(<CURL>) {
      chomp;
      $reply .= $_;
    }
    
    #print "reply : \n$reply\n";
    
    $reply =~ m/<p:properties [~>]*>/g;
    my @params = ();
    #while ( $reply !~ m/[^<]*<\/p:properties>/g ) {
    while (1) {
#       if ( $reply =~ m/[^\/]p:([^\s]+) xsi:type=\"(soapenc:Struct)\">(.*)<\/p:$1/g ) {
#           printf "match $1 $2 $3\n";
#       } els
if ( $reply !~ m/[^\/]p:([^\s]+) xsi:type=\"([^\"]+)\">([^<]*)<\/p:/g ) {
           last;
       }
       #print "$1 $2 $3\n";
       my $para = { 'name' => "$1", 'type' => "$2", 'value' => "$3" };
       push @params, $para;
    }
    return \@params;
}
