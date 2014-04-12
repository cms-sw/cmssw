    #!/usr/local/bin/perl -w
    # File:  go2www
    # This Perl program in classic programming style changes
    # the string "gopher" to "World Wide Web" in all files
    # specified on the command line.
    # 19950926 gkj
    $original=$ARGV[0];
    $replacement=$ARGV[1];
    $nchanges = 0;
    # The input record separator is defined by Perl global
    # variable $/.  It can be anything, including multiple
    # characters.  Normally it is "\n", newline.  Here, we
    # say there is no record separator, so the whole file
    # is read as one long record, newlines included.
    undef $/;

    print STDERR "Changing $ARGV[0] to $ARGV[1]\n";

    # Suppose this program was invoked with the command
    #     go2www ax.html  big.basket.html  candle.html
    # Then builtin list @ARGV would contain three elments 
    # ('ax.html', 'big.basket.html', 'candle.html')
    # These could be accessed as $ARGV[0] $ARGV[1] $ARGV[2] 

    foreach $file (@ARGV[2..$#ARGV]) {
        if (! open(INPUT,"<$file") ) {
            print STDERR "Can't open input file $bakfile\n";
            next;
        }

        # Read input file as one long record.
        $data=<INPUT>;
        close INPUT;

        if ($data =~ s/$original/$replacement/g) {
            $bakfile = "\.pl_backups/$file.bak";
            # Abort if can't backup original or output.
            if (! rename($file,$bakfile)) {
                die "Can't rename $file $!";
            }
            if (! open(OUTPUT,">$file") ) {
                die "Can't open output file $file\n";
            }
            print OUTPUT $data;
            close OUTPUT;
            print STDERR "$file changed\n";
            $nchanges++;
        }

        else {  print STDERR "$file not changed\n"; }
    }
    print STDERR "$nchanges files changed.\n";
    exit(0);
