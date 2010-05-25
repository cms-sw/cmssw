#!/usr/bin/perl

use Getopt::Long;
my ($myfile);
GetOptions( 
		"myfile:s" => \$myfile,
		"user:s" => \$user,
		"password:s" => \$password,
		"db:s" => \$db,

	  );
if(!$myfile or !$user or !$password or !$db){
	die "perl update_tag.pl -myfile list_tag.txt -user CMS_COND_STRIP -db cms_orcoff_int2r -password SSWDC3MCAI8HQHTC\n";
}

unless(-e $myfile){die "Il file specificato doesn't exit\n"; }

open (TAG,$myfile);
while ($item_tag = <TAG>){
	chomp($item_tag);
	$item_tag_mod=$item_tag."_mc";
#print "select name from metadata where name='$item_tag';\n";
#@pass=`echo "select name from metadata where name='$item_tag';" | sqlplus CMS_COND_STRIP\@cms_orcoff_int2r/SSWDC3MCAI8HQHTC`;
	@pass=`echo "update metadata set name='$item_tag\_mc' where name='$item_tag';" | sqlplus $user\@$db/$password`;
#print "echo \"update metadata set name='$item_tag\_mc' where name='$item_tag';\" | sqlplus CMS_COND_STRIP\@cms_orcoff_int2r/SSWDC3MCAI8HQHTC\n";
	print @pass;
}
close TAG;
