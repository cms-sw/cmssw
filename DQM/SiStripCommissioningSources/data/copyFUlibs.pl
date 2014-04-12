#!/usr/bin/env perl
use DirHandle;
use File::Copy;
$oldfile = @ARGV[0];
$destination = @ARGV[1];
open( INFILE, "$oldfile") || die "ERROR::could not open input file $oldfile";
while( $record = <INFILE> )
{
    if($record =~ /LD_LIBRARY_PATH=/){
	print "$record\n";
	@tok1 = split(/=\"/, $record);
	print "@tok1\n";
	#pop(@tok1);
	#pop(@tok1);
	print "this is tok1\n";
	print "@tok1[1]\n";
	mkdir ($destination);
	opendir(LIBS,$destination);
	@tokens = split(/\:/,@tok1[1]);
	while($#tokens != -1)
	{
	    if(@tokens[$#tokens] =~ /\$/)
	    {}
	    else
	    {
		print "token $#tokens @tokens[$#tokens]\n";
		opendir( DIR, @tokens[$#tokens]);
		@dots = grep {(/\.so/ || /\.a/ || /\.0/)} readdir(DIR);
		while($#dots != -1)
		{
		    $foundlink = readlink "$destination\/@dots[$#dots]";
#		    print "$foundlink\n";
		    system `rm $destination\/@dots[$#dots]` if $foundlink;
#		    symlink("@tokens[$#tokens]\/@dots[$#dots]","$destination\/@dots[$#dots]");
		    copy("@tokens[$#tokens]\/@dots[$#dots]","$destination\/@dots[$#dots]");
		    pop(@dots);
		}
		print "@dots\n";
	    }
	    pop(@tokens);
	}
    }
}
