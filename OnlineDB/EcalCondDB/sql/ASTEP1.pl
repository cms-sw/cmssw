#!/usr/bin/env perl
#!/usr/bin/env perl

use Getopt::Long;

$year;
GetOptions('year=n' => \$year);
if ((!$year) || (length $year > 2)) {
    print "Usage: ASTEP1.pl --year [year]\n";
    print "       [year] must be two digits\n";
    exit 0;
}

$cyear = $year;
if ($year < 10) {
    $cyear = "0" . $year;
}

# first of all get the primary key for each table
open IN, "pks.data";
@buffer = <IN>;
close IN;

%pk;
foreach $line (@buffer) {
    chomp $line;
    $line =~ s/ +$//;
    ($table, $key) = split / +/, $line;
    $pk{$table} = $key;
}

print "--> READ PRIMARY KEYS\n";

# then get the size of the tables and the maximum IOV value
open IN, "partition.data";
@buffer = <IN>;
close IN;

%iov;
foreach $line (@buffer) {
    chomp $line;
    $line =~ s/ +$//;
    ($table, $key, $size, $iovs) = split / +/, $line;
    if ($iovs == '') {
	$iovs = 1;
    } else {
	$iovs += 1;
    }
    if ((exists $pk{$table}) && ($pk{$table} == $key)) {
	$iov{$table} = $iovs;
    }
}

print "--> READ TABLE SIZE\n";

# then gets info on indexes

open IN, "indexes.data";
@buffer = <IN>;
close IN;

%index;
$last_index_name = "";
foreach $line (@buffer) {
    chomp $line;
    $line =~ s/ +$//;
    ($index_name, $table, $column) = split / +/, $line;
    if (exists $pk{$table}) {
	$indx = "  CREATE INDEX " . $index_name . " ON " . $table .
	    " (" . $column .");\n";
# treat index with multiple columns
	if ($last_index_name !~ m/^$index_name$/) {
	    $last_index_name = $index_name;
	} else {
	    $regexp = ", $column);\n";
	    $index{$table} =~ s/..$/$regexp/;
	    $indx = "";
	}
	if (!(exists $index{$table})) {
	    $index{$table} = $indx;
	} else {
	    $index{$table} .= $indx;
	}
    }
}

# now create sql statements to copy tables

open IN, "recreate.sql";
@buffer = <IN>;
close IN;

$start = 0;
$table_name = "";
$sql = "";
$sql2 = "";
$constraints = "";
$extrasql .= "  PCTFREE 10 PCTUSED 40 INITRANS 1 MAXTRANS 255\n" .
    "  STORAGE(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645\n" .
    "  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1 BUFFER_POOL DEFAULT)";
open OUT, ">copytables.sql";
@tables;
@references;
%deferred;
$drop = "";
foreach $line (@buffer) {
    next unless $line !~ m/^ +$/; # skip blank lines
    $line =~ s/ +$//;
    if (length($line) > 80) {
	chomp $line;
    }
    if ($line =~ m/CREATE TABLE/) {
	$table_name = $line;
	chomp $table_name;
	$table_name =~ s/\"$//;
	$table_name =~ s/.*\"//;
	if (length($table_name) > 27) {
	    print "WARNING: TABLE $table_name has a name too long\n";
	}
	@references = ();
	if (exists $pk{$table_name}) {
	    $ic = length($line) - 3;
	    $c = nextChar($line, $ic);
	    $line =~ s/(.).\" *$/\1$c\"/;
	}
	$start = 1;
    }
    if ($line =~ m/REFERENCES/) {
	$referredTable = $line;
	($part1, $part2) = split / \(/, $referredTable;
	$refT = $part1;
	$refT =~ s/\"$//;
	$refT =~ s/.*\"//;
	if (exists $pk{$refT}) {
	    $ic = length($part1) - 2;
	    $c = nextChar($part1, $ic);
	    $part1 =~ s/..$/$c\"/;
	}
	$referredTable = $part1;
	$referredTable =~ s/.*\"([^\"]+)\"/\1/;
	push @references, $referredTable;
	$line = $part1 . " (" . $part2;
    }
    if ($line =~ m/CONSTRAINT/) {
	@tmp = split / +/, $line;
	$ic = 1;
#	$ic = length($tmp[2]) + length($tmp[1]) + length($tmp[0]) + 3 - 3;
	$c = nextChar($tmp[2], $ic);
#	$line =~ s/(CONSTRAINT \"[^ ]+).\"/\1$c\"/;
	$line =~ s/(CONSTRAINT \")./\1$c/;
	$old_const_name = $tmp[2];
	$old_const_name =~ s/\"//g;
	$const_name = $old_const_name;
#	$const_name =~ s/(.*).$/\1$c/;
	$const_name =~ s/^./$c/;
	$constraints .= "  ALTER TABLE " . $table_name . " RENAME CONSTRAINT " .
	    $const_name . " TO " . $old_const_name . ";\n";

#
# take into account automatic indexes
#
	if (($line =~ m/PRIMARY KEY/) || ($line =~ m/UNIQUE/)) {
	    $constraints .= "  ALTER INDEX " . $const_name . " RENAME TO " . $old_const_name . ";\n";
# drop them from the list of indexes
	    $regexp = "  CREATE INDEX $old_const_name ON $table_name [^\n]+";
	    $index{$table_name} =~ s/$regexp/  /g;
	}
    }
    @try = split / +/, $line;
    $nTry = @try;
    if (($line =~ /TABLESPACE/) && ($nTry == 3) && ($start == 1)) {
	$sql .= $line;
	if (length $table_name > 27) {
	    print "ERROR: $table_name name is too long!!!\n";
	}
	$sql .= "  PARTITION BY RANGE (\"" . $pk{$table_name} . "\")\n";
	$sql .= "  (PARTITION \"" . $table_name . "_" . $cyear . 
	    "\" VALUES LESS THAN (" . $iov{$table_name} . ")\n" . $extrasql . ",\n";
	$sql .= "  PARTITION \"" . $table_name . "_0" . 
	    "\" VALUES LESS THAN (MAXVALUE)\n" . $extrasql . "\n";
	$sql .= "  );\n";
	$newtable_name = $table_name;
	if (exists $pk{$table_name}) {
	    $c = nextChar($newtable_name, length($newtable_name) - 1);
	    $newtable_name =~ s/(.*).$/\1$c/;
	}
	$sql .= "  INSERT INTO " . $newtable_name . 
	    " (SELECT * FROM " . $table_name . ");\n";
	$sql .= "\n";
	$sql2 = "  DROP TABLE $table_name CASCADE CONSTRAINTS;\n";
	$sql2 .= "  RENAME " . $newtable_name . " TO " . $table_name . ";\n";
	$sql2 .= $constraints;
# correct index creation statements substituting special characters with newlines
	$sql2 .= $index{$table_name};
	$sql2 .= "\n";
# check if we need to defer this definition since it depends on other tables
	$toDefer = 0;
	if (exists $pk{$table_name}) {
	    foreach $t (@references) {
		@alreadyDef = grep {/^$t$/} @tables;
		$n = @alreadyDef;
		if ($n == 0) {
		    $toDefer = 1;
		    if (exists $deferred{$newtable_name}) {
			$deferred{$newtable_name} .= "#" . $t;
		    } else {
			$deferred{$newtable_name} = $t;
		    }
		}
	    }
	}
#	if ($toDefer) {
#	    print "DEFERRING CREATION OF TABLE $newtable_name\n";
#	    $deferred{$newtable_name} .= "#" . $sql . "#" . $sql2;
#	} else {
#	    if (exists $pk{$table_name}) {
#		print OUT "  PROMPT Creating $newtable_name\n";
#		print OUT $sql;
#		$drop .= $sql2;
#		push @tables, $newtable_name;
#		print "CREATING TABLE $newtable_name\n";
#	    } else {
#		push @tables, $table_name;
#		print "KEEPING  TABLE $table_name\n";
#	    }
#	}
	if (exists $pk{$table_name}) {
	    if ($toDefer) {
		print "DEFERRING CREATION OF TABLE $newtable_name\n";
		$deferred{$newtable_name} .= "#" . $sql . "#" . $sql2;
	    } else {
		print OUT "  PROMPT Creating $newtable_name\n";
		print OUT $sql;
		$drop .= $sql2;
		push @tables, $newtable_name;
		print "CREATING TABLE $newtable_name\n";
	    } 
	} else {
	    push @tables, $table_name;
	    print "KEEPING  TABLE $table_name\n";
	}
	$start = 0;
	$sql = "";
	$constraints = "";
    }
    if ($start == 1) {
	$sql .= $line;
    }
}

print "========================================================\n";
print "List of already defined table\n";
foreach $t (@tables) {
    print "$t\n";
}
print "========================================================\n";

# print remaining statements
$loop = 1;
while ($loop != 0) {
    @remKeys = keys %deferred;
    foreach $k (@remKeys) {
	@array = split/\#/,$deferred{$k};
	$lar = @array;
	$sql = $array[$lar - 2];
	$sql2 = $array[$lar - 1];
#	@buf = split /\n/, $sql;
#	@required = ();
#	foreach $line (@buf) {
#	    $table;
#	    if ($line =~ m/CREATE TABLE/) {
#		$table = $line;
#		$table =~ s/.$//;
#		$table =~ s/.*\"//;
#	    }
#	}
	pop @array;
	pop @array;
	$nRequired = @array;
	$nFound = 0;
	print "$k requires the definition of $nRequired tables\n";
	foreach $a (@array) {
	    print "$k: checking $a...";
	    @gres = grep {/^$a$/} @tables;
	    $ngres = @gres;
	    if ($ngres != 0) {
		print "Found! ";
		foreach $tt (@gres) {
		    print "$tt ";
		}
		print "\n";
	    } else {
		print "\n";
	    }
	    $nFound += $ngres;
	}
	if ($nFound == $nRequired) {
	    print "$k: TABLE DUMPED\n";
	    print OUT "  PROMPT Creating $k\n";
	    print OUT $sql;
	    $drop .= $sql2;
	    delete $deferred{$k};
	    push @tables, $k;
	}
    }
    @remKeys = keys %deferred;
    $loop = @remKeys;
    print "----------- Remaining keys $loop\n\n";
}

close OUT;

open OUT, ">dropoldtables.sql";
print OUT $drop;
close OUT;
exit 0;

sub nextChar() {
    my $line = @_[0];
    my $ic = @_[1];
    my @chars = split(//, $line);
    my $r = chr(ord($chars[$ic]) + 1);
    return $r;
}

sub prevChar() {
    my $line = @_[0];
    my $ic = @_[1];
    my @chars = split(//, $line);
    my $r = chr(ord($chars[$ic]) - 1);
}
