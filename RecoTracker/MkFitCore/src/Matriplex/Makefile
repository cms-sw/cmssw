all: auto

auto: std_sym intr_sym

clean:
	rm -f *.ah

distclean: clean

# ---------------------------------------------------------------- #

std_sym:  std_sym_3x3.ah std_sym_6x6.ah

intr_sym: intr_sym_3x3.ah intr_sym_6x6.ah 


# ================================================================ #

GM := ./gen_mul.pl

std_sym_3x3.ah:
	${GM} "mult_sym(3);" > $@

std_sym_6x6.ah:
	${GM} "mult_sym(6);" > $@

intr_sym_3x3.ah:
	${GM} "mult_sym_fma_intrinsic(3);" > $@

intr_sym_6x6.ah:
	${GM} "mult_sym_fma_intrinsic(6);" > $@
