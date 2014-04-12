#!/usr/bin/env perl
                                                                                                                             
die "Usage sendCmdToApp.pl host port cmdFile\n" if @ARGV != 3;
                                                                                                                             
$host    = $ARGV[0];
$port    = $ARGV[1];
$cmdFile = $ARGV[2];
                                                                                                                             
die "The file \"$cmdFile\" does not exist\n" unless -e $cmdFile;

$curlCmd = "curl --stderr /dev/null -H \"Content-Type: text/xml\" -H \"Content-Description: SOAP Message\" -H \"SOAPAction: urn:xdaq-application:lid=0\" http://$host:$port -d \@$cmdFile";  
                                                                                                                           
open CURL, "$curlCmd|";
                                                                                                                             
print "Sending command to executive $host:$port ";
                                                                                                                             
while(<CURL>) {
    chomp;
    $soapReply .= $_;
}
                                                                                                                             
if($soapReply =~ m/Response/) {
    print "OK\n";
    exit 0;
} elsif($soapReply =~ m/Fault/) {
    print "FAULT\n";
    print "$soapReply\n";
    exit 1;
} elsif($soapReply eq "") {
    print "NONE\n";
    exit 1;
} else {
    print "UNKNOWN response\n";
    print "$soapReply\n";
    exit 1;
}
