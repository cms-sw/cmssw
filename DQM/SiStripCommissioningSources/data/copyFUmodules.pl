#!/usr/bin/env perl
use DirHandle;
use File::Copy;
$oldfile = @ARGV[0];
$destination = @ARGV[1];
open( INFILE, "$oldfile") || die "ERROR::could not open input file $oldfile";
while( $record = <INFILE> )
{

    if($record =~ /SEAL_PLUGINS=/){
	@tok1 = split(/=\"/, $record);
	print "@tok1\n";
	#pop(@tok1);
	#pop(@tok1);
	@tok1[1] =~ s/\";\n//;
	print "going to symlink from\n";
	print "@tok1[1]\n";
	mkdir ($destination);
	opendir(MODULES,$destination);
	@tokens = split(/\:/,@tok1[1]);
	while($#tokens != -1)
	{
	    if(@tokens[$#tokens] =~ /\$/)
	    {}
	    else
	    {
		print "token $#tokens @tokens[$#tokens]\n";
		opendir(DIR, @tokens[$#tokens]);
		@dots = grep {(! /^\./) && (/\.reg/)} readdir(DIR);
		print "found $#dots \n";
		while($#dots != -1)
		{
		    print " symlinking @tokens[$#tokens]\/@dots[$#dots]\n";
		    copy("@tokens[$#tokens]\/@dots[$#dots]","$destination\/@dots[$#dots]");
		    pop(@dots);
		}
		print "@dots\n";
	    }
	    pop(@tokens);
	}
    }
}
