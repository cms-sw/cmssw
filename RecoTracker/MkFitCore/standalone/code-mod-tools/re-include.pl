#!/usr/bin/perl

$RUNDIR = '/data2/matevz/CMSSW_12_2_0_pre2/src';

$pwd = `pwd`; chomp $pwd; die "Has to be run in $RUNDIR" unless $pwd eq $RUNDIR;

# all files: `find RecoTracker -name \*.h -or -name \*.cc -or -name \*.icc -or -name \*.acc`;

### Setup headers, short-headers and header map. Exclude standalone/

@headers = grep { chomp; $_ !~ m!/(?:standalone)/!; } `find RecoTracker -name \*.h`;

@sheaders = map { my $a = $_; $a = $1 if $a =~ m!.*/([^/]+)$!; $a; } @headers;

$NH = scalar(@headers);

%hmap = ();
for (my $i=0; $i<$NH; ++$i) { $hmap{$sheaders[$i]} = $headers[$i];}


# Setup files to process, filter out stuff we don't want to touch.

@files = grep { chomp; $_ !~ m!/(?:attic|dusty-chest)/!; } `find RecoTracker -name \*.h -or -name \*.cc`;

print "HEADERS:\n";
for (my $i=0; $i<$NH; ++$i) { print "  ", $headers[$i], "  -->  ", $sheaders[$i], "\n"; }
print "\n\n";
print "FILES:\n", join("\n", @files);
print "\n\n";


################################################################

for my $file (@files)
{
    open F, $file;
    @lines = <F>;
    close F;

    # print $file,"\n",$f,"\n\n";

    my $insrc = $file =~ m!^(.*/(?:src|plugins)(?:/.*)?)/[^/]+$!;
    $insrc = $1 if $insrc;

    print "Processing file $file, N_lines = ", scalar(@lines), ", in_src = ", $insrc, "\n";

    my $changed = 0;

    for my $l (@lines)
    {
        if ($l =~ m!^#include\s+"(.*)"\s*!)
        {
            my $sh = $1; $sh = $1 if $sh =~ m!.*/([^/]+)$!;

            my $have = exists $hmap{$sh};

            my $line_to_print = $l;
            chomp $line_to_print;
            print "Found includeline $line_to_print -- $sh --> $hmap{$sh}\n";

            if ($have)
            {
                # replace the line ... but first check if these are in the same src/ directory.
                my $full_inc = $hmap{$sh};
                if ($insrc && ($full_inc =~ m!^${insrc}!))
                {
                    $full_inc =~ s!^${insrc}/(.*)!$1!;
                    print "   QQQQQQQ File and include in the same src/ --> shortening to ${full_inc}\n";
                }

                $l = "#include \"${full_inc}\"\n";
                $changed = 1;

                print "  new line is $l";
            }
        }
    }

    if ($changed)
    {
        open F, ">$file";
        print F @lines;
        close F;
    }
}
