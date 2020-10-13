webpackHotUpdate_N_E("pages/index",{

/***/ "./components/Nav.tsx":
/*!****************************!*\
  !*** ./components/Nav.tsx ***!
  \****************************/
/*! exports provided: Nav, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "Nav", function() { return Nav; });
/* harmony import */ var _babel_runtime_helpers_esm_extends__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @babel/runtime/helpers/esm/extends */ "./node_modules/@babel/runtime/helpers/esm/extends.js");
/* harmony import */ var _babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @babel/runtime/helpers/esm/slicedToArray */ "./node_modules/@babel/runtime/helpers/esm/slicedToArray.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ./styledComponents */ "./components/styledComponents.ts");
/* harmony import */ var _searchButton__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ./searchButton */ "./components/searchButton.tsx");
/* harmony import */ var _helpButton__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ./helpButton */ "./components/helpButton.tsx");



var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/components/Nav.tsx",
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_2___default.a.createElement;





var Nav = function Nav(_ref) {
  _s();

  var initial_search_run_number = _ref.initial_search_run_number,
      initial_search_dataset_name = _ref.initial_search_dataset_name,
      setRunNumber = _ref.setRunNumber,
      setDatasetName = _ref.setDatasetName,
      handler = _ref.handler,
      type = _ref.type,
      defaultRunNumber = _ref.defaultRunNumber,
      defaultDatasetName = _ref.defaultDatasetName;

  var _Form$useForm = antd__WEBPACK_IMPORTED_MODULE_3__["Form"].useForm(),
      _Form$useForm2 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__["default"])(_Form$useForm, 1),
      form = _Form$useForm2[0];

  var _useState = Object(react__WEBPACK_IMPORTED_MODULE_2__["useState"])(initial_search_run_number || ''),
      form_search_run_number = _useState[0],
      setFormRunNumber = _useState[1];

  var _useState2 = Object(react__WEBPACK_IMPORTED_MODULE_2__["useState"])(initial_search_dataset_name || ''),
      form_search_dataset_name = _useState2[0],
      setFormDatasetName = _useState2[1]; // We have to wait for changin initial_search_run_number and initial_search_dataset_name coming from query, because the first render they are undefined and therefore the initialValues doesn't grab them


  Object(react__WEBPACK_IMPORTED_MODULE_2__["useEffect"])(function () {
    form.resetFields();
    setFormRunNumber(initial_search_run_number || '');
    setFormDatasetName(initial_search_dataset_name || '');
  }, [initial_search_run_number, initial_search_dataset_name, form]);
  var layout = {
    labelCol: {
      span: 8
    },
    wrapperCol: {
      span: 16
    }
  };
  var tailLayout = {
    wrapperCol: {
      offset: 0,
      span: 4
    }
  };
  return __jsx("div", {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 54,
      columnNumber: 5
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["CustomForm"], Object(_babel_runtime_helpers_esm_extends__WEBPACK_IMPORTED_MODULE_0__["default"])({
    form: form,
    layout: 'inline',
    justifycontent: "center",
    width: "max-content"
  }, layout, {
    name: "search_form".concat(type),
    className: "fieldLabel",
    initialValues: {
      run_number: initial_search_run_number,
      dataset_name: initial_search_dataset_name
    },
    onFinish: function onFinish() {
      setRunNumber && setRunNumber(form_search_run_number);
      setDatasetName && setDatasetName(form_search_dataset_name);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 55,
      columnNumber: 7
    }
  }), __jsx(antd__WEBPACK_IMPORTED_MODULE_3__["Form"].Item, Object(_babel_runtime_helpers_esm_extends__WEBPACK_IMPORTED_MODULE_0__["default"])({}, tailLayout, {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 72,
      columnNumber: 9
    }
  }), __jsx(_helpButton__WEBPACK_IMPORTED_MODULE_6__["QuestionButton"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 73,
      columnNumber: 11
    }
  })), __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledFormItem"], {
    name: "run_number",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 75,
      columnNumber: 9
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledInput"], {
    id: "run_number",
    onChange: function onChange(e) {
      return setFormRunNumber(e.target.value);
    },
    placeholder: "Enter run number",
    type: "text",
    name: "run_number",
    value: defaultRunNumber,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 76,
      columnNumber: 11
    }
  })), __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledFormItem"], {
    name: "dataset_name",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 87,
      columnNumber: 9
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledInput"], {
    id: "dataset_name",
    placeholder: "Enter dataset name",
    onChange: function onChange(e) {
      return setFormDatasetName(e.target.value);
    },
    type: "text",
    value: defaultDatasetName,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 88,
      columnNumber: 11
    }
  })), __jsx(antd__WEBPACK_IMPORTED_MODULE_3__["Form"].Item, Object(_babel_runtime_helpers_esm_extends__WEBPACK_IMPORTED_MODULE_0__["default"])({}, tailLayout, {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 98,
      columnNumber: 9
    }
  }), __jsx(_searchButton__WEBPACK_IMPORTED_MODULE_5__["SearchButton"], {
    onClick: function onClick() {
      return handler(form_search_run_number, form_search_dataset_name);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 99,
      columnNumber: 11
    }
  }))));
};

_s(Nav, "d/o1hn25bH6EF0LAvbTEx8d/DOY=", false, function () {
  return [antd__WEBPACK_IMPORTED_MODULE_3__["Form"].useForm];
});

_c = Nav;
/* harmony default export */ __webpack_exports__["default"] = (Nav);

var _c;

$RefreshReg$(_c, "Nav");

;
    var _a, _b;
    // Legacy CSS implementations will `eval` browser code in a Node.js context
    // to extract CSS. For backwards compatibility, we need to check we're in a
    // browser context before continuing.
    if (typeof self !== 'undefined' &&
        // AMP / No-JS mode does not inject these helpers:
        '$RefreshHelpers$' in self) {
        var currentExports = module.__proto__.exports;
        var prevExports = (_b = (_a = module.hot.data) === null || _a === void 0 ? void 0 : _a.prevExports) !== null && _b !== void 0 ? _b : null;
        // This cannot happen in MainTemplate because the exports mismatch between
        // templating and execution.
        self.$RefreshHelpers$.registerExportsForReactRefresh(currentExports, module.i);
        // A module can be accepted automatically based on its exports, e.g. when
        // it is a Refresh Boundary.
        if (self.$RefreshHelpers$.isReactRefreshBoundary(currentExports)) {
            // Save the previous exports on update so we can compare the boundary
            // signatures.
            module.hot.dispose(function (data) {
                data.prevExports = currentExports;
            });
            // Unconditionally accept an update to this module, we'll check if it's
            // still a Refresh Boundary later.
            module.hot.accept();
            // This field is set when the previous version of this module was a
            // Refresh Boundary, letting us know we need to check for invalidation or
            // enqueue an update.
            if (prevExports !== null) {
                // A boundary can become ineligible if its exports are incompatible
                // with the previous exports.
                //
                // For example, if you add/remove/change exports, we'll want to
                // re-execute the importing modules, and force those components to
                // re-render. Similarly, if you convert a class component to a
                // function, we want to invalidate the boundary.
                if (self.$RefreshHelpers$.shouldInvalidateReactRefreshBoundary(prevExports, currentExports)) {
                    module.hot.invalidate();
                }
                else {
                    self.$RefreshHelpers$.scheduleUpdate();
                }
            }
        }
        else {
            // Since we just executed the code for the module, it's possible that the
            // new exports made it ineligible for being a boundary.
            // We only care about the case when we were _previously_ a boundary,
            // because we already accepted this update (accidental side effect).
            var isNoLongerABoundary = prevExports !== null;
            if (isNoLongerABoundary) {
                module.hot.invalidate();
            }
        }
    }

/* WEBPACK VAR INJECTION */}.call(this, __webpack_require__(/*! ./../node_modules/webpack/buildin/harmony-module.js */ "./node_modules/webpack/buildin/harmony-module.js")(module)))

/***/ })

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9OYXYudHN4Il0sIm5hbWVzIjpbIk5hdiIsImluaXRpYWxfc2VhcmNoX3J1bl9udW1iZXIiLCJpbml0aWFsX3NlYXJjaF9kYXRhc2V0X25hbWUiLCJzZXRSdW5OdW1iZXIiLCJzZXREYXRhc2V0TmFtZSIsImhhbmRsZXIiLCJ0eXBlIiwiZGVmYXVsdFJ1bk51bWJlciIsImRlZmF1bHREYXRhc2V0TmFtZSIsIkZvcm0iLCJ1c2VGb3JtIiwiZm9ybSIsInVzZVN0YXRlIiwiZm9ybV9zZWFyY2hfcnVuX251bWJlciIsInNldEZvcm1SdW5OdW1iZXIiLCJmb3JtX3NlYXJjaF9kYXRhc2V0X25hbWUiLCJzZXRGb3JtRGF0YXNldE5hbWUiLCJ1c2VFZmZlY3QiLCJyZXNldEZpZWxkcyIsImxheW91dCIsImxhYmVsQ29sIiwic3BhbiIsIndyYXBwZXJDb2wiLCJ0YWlsTGF5b3V0Iiwib2Zmc2V0IiwicnVuX251bWJlciIsImRhdGFzZXRfbmFtZSIsImUiLCJ0YXJnZXQiLCJ2YWx1ZSJdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBO0FBQ0E7QUFFQTtBQUNBO0FBQ0E7QUFjTyxJQUFNQSxHQUFHLEdBQUcsU0FBTkEsR0FBTSxPQVNIO0FBQUE7O0FBQUEsTUFSZEMseUJBUWMsUUFSZEEseUJBUWM7QUFBQSxNQVBkQywyQkFPYyxRQVBkQSwyQkFPYztBQUFBLE1BTmRDLFlBTWMsUUFOZEEsWUFNYztBQUFBLE1BTGRDLGNBS2MsUUFMZEEsY0FLYztBQUFBLE1BSmRDLE9BSWMsUUFKZEEsT0FJYztBQUFBLE1BSGRDLElBR2MsUUFIZEEsSUFHYztBQUFBLE1BRmRDLGdCQUVjLFFBRmRBLGdCQUVjO0FBQUEsTUFEZEMsa0JBQ2MsUUFEZEEsa0JBQ2M7O0FBQUEsc0JBQ0NDLHlDQUFJLENBQUNDLE9BQUwsRUFERDtBQUFBO0FBQUEsTUFDUEMsSUFETzs7QUFBQSxrQkFFcUNDLHNEQUFRLENB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QWY7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxNQUNFLE1BQUMsMERBQUQ7QUFDRSxXQUFPLEVBQUU7QUFBQSxhQUNQbEIsT0FBTyxDQUFDUSxzQkFBRCxFQUF5QkUsd0JBQXpCLENBREE7QUFBQSxLQURYO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFERixDQTNDRixDQURGLENBREY7QUF1REQsQ0F4Rk07O0dBQU1mLEc7VUFVSVMseUNBQUksQ0FBQ0MsTzs7O0tBVlRWLEc7QUEwRkVBLGtFQUFmIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4LjMyY2NmOTlmZjkzYzg3MjhkYjVkLmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgUmVhY3QsIHsgQ2hhbmdlRXZlbnQsIERpc3BhdGNoLCB1c2VFZmZlY3QsIHVzZVN0YXRlIH0gZnJvbSAncmVhY3QnO1xuaW1wb3J0IHsgRm9ybSB9IGZyb20gJ2FudGQnO1xuXG5pbXBvcnQgeyBTdHlsZWRGb3JtSXRlbSwgU3R5bGVkSW5wdXQsIEN1c3RvbUZvcm0gfSBmcm9tICcuL3N0eWxlZENvbXBvbmVudHMnO1xuaW1wb3J0IHsgU2VhcmNoQnV0dG9uIH0gZnJvbSAnLi9zZWFyY2hCdXR0b24nO1xuaW1wb3J0IHsgUXVlc3Rpb25CdXR0b24gfSBmcm9tICcuL2hlbHBCdXR0b24nO1xuaW1wb3J0IHsgZnVuY3Rpb25zX2NvbmZpZyB9IGZyb20gJy4uL2NvbmZpZy9jb25maWcnO1xuXG5pbnRlcmZhY2UgTmF2UHJvcHMge1xuICBzZXRSdW5OdW1iZXI/OiBEaXNwYXRjaDxhbnk+O1xuICBzZXREYXRhc2V0TmFtZT86IERpc3BhdGNoPGFueT47XG4gIGluaXRpYWxfc2VhcmNoX3J1bl9udW1iZXI/OiBzdHJpbmc7XG4gIGluaXRpYWxfc2VhcmNoX2RhdGFzZXRfbmFtZT86IHN0cmluZztcbiAgaGFuZGxlcihzZWFyY2hfYnlfcnVuX251bWJlcjogc3RyaW5nLCBzZWFyY2hfYnlfZGF0YXNldF9uYW1lOiBzdHJpbmcpOiB2b2lkO1xuICB0eXBlOiBzdHJpbmc7XG4gIGRlZmF1bHRSdW5OdW1iZXI/OiB1bmRlZmluZWQgfCBzdHJpbmc7XG4gIGRlZmF1bHREYXRhc2V0TmFtZT86IHN0cmluZyB8IHVuZGVmaW5lZDtcbn1cblxuZXhwb3J0IGNvbnN0IE5hdiA9ICh7XG4gIGluaXRpYWxfc2VhcmNoX3J1bl9udW1iZXIsXG4gIGluaXRpYWxfc2VhcmNoX2RhdGFzZXRfbmFtZSxcbiAgc2V0UnVuTnVtYmVyLFxuICBzZXREYXRhc2V0TmFtZSxcbiAgaGFuZGxlcixcbiAgdHlwZSxcbiAgZGVmYXVsdFJ1bk51bWJlcixcbiAgZGVmYXVsdERhdGFzZXROYW1lLFxufTogTmF2UHJvcHMpID0+IHtcbiAgY29uc3QgW2Zvcm1dID0gRm9ybS51c2VGb3JtKCk7XG4gIGNvbnN0IFtmb3JtX3NlYXJjaF9ydW5fbnVtYmVyLCBzZXRGb3JtUnVuTnVtYmVyXSA9IHVzZVN0YXRlKFxuICAgIGluaXRpYWxfc2VhcmNoX3J1bl9udW1iZXIgfHwgJydcbiAgKTtcbiAgY29uc3QgW2Zvcm1fc2VhcmNoX2RhdGFzZXRfbmFtZSwgc2V0Rm9ybURhdGFzZXROYW1lXSA9IHVzZVN0YXRlKFxuICAgIGluaXRpYWxfc2VhcmNoX2RhdGFzZXRfbmFtZSB8fCAnJ1xuICApO1xuXG4gIC8vIFdlIGhhdmUgdG8gd2FpdCBmb3IgY2hhbmdpbiBpbml0aWFsX3NlYXJjaF9ydW5fbnVtYmVyIGFuZCBpbml0aWFsX3NlYXJjaF9kYXRhc2V0X25hbWUgY29taW5nIGZyb20gcXVlcnksIGJlY2F1c2UgdGhlIGZpcnN0IHJlbmRlciB0aGV5IGFyZSB1bmRlZmluZWQgYW5kIHRoZXJlZm9yZSB0aGUgaW5pdGlhbFZhbHVlcyBkb2Vzbid0IGdyYWIgdGhlbVxuICB1c2VFZmZlY3QoKCkgPT4ge1xuICAgIGZvcm0ucmVzZXRGaWVsZHMoKTtcbiAgICBzZXRGb3JtUnVuTnVtYmVyKGluaXRpYWxfc2VhcmNoX3J1bl9udW1iZXIgfHwgJycpO1xuICAgIHNldEZvcm1EYXRhc2V0TmFtZShpbml0aWFsX3NlYXJjaF9kYXRhc2V0X25hbWUgfHwgJycpO1xuICB9LCBbaW5pdGlhbF9zZWFyY2hfcnVuX251bWJlciwgaW5pdGlhbF9zZWFyY2hfZGF0YXNldF9uYW1lLCBmb3JtXSk7XG5cbiAgY29uc3QgbGF5b3V0ID0ge1xuICAgIGxhYmVsQ29sOiB7IHNwYW46IDggfSxcbiAgICB3cmFwcGVyQ29sOiB7IHNwYW46IDE2IH0sXG4gIH07XG4gIGNvbnN0IHRhaWxMYXlvdXQgPSB7XG4gICAgd3JhcHBlckNvbDogeyBvZmZzZXQ6IDAsIHNwYW46IDQgfSxcbiAgfTtcblxuICByZXR1cm4gKFxuICAgIDxkaXY+XG4gICAgICA8Q3VzdG9tRm9ybVxuICAgICAgICBmb3JtPXtmb3JtfVxuICAgICAgICBsYXlvdXQ9eydpbmxpbmUnfVxuICAgICAgICBqdXN0aWZ5Y29udGVudD1cImNlbnRlclwiXG4gICAgICAgIHdpZHRoPVwibWF4LWNvbnRlbnRcIlxuICAgICAgICB7Li4ubGF5b3V0fVxuICAgICAgICBuYW1lPXtgc2VhcmNoX2Zvcm0ke3R5cGV9YH1cbiAgICAgICAgY2xhc3NOYW1lPVwiZmllbGRMYWJlbFwiXG4gICAgICAgIGluaXRpYWxWYWx1ZXM9e3tcbiAgICAgICAgICBydW5fbnVtYmVyOiBpbml0aWFsX3NlYXJjaF9ydW5fbnVtYmVyLFxuICAgICAgICAgIGRhdGFzZXRfbmFtZTogaW5pdGlhbF9zZWFyY2hfZGF0YXNldF9uYW1lLFxuICAgICAgICB9fVxuICAgICAgICBvbkZpbmlzaD17KCkgPT4ge1xuICAgICAgICAgIHNldFJ1bk51bWJlciAmJiBzZXRSdW5OdW1iZXIoZm9ybV9zZWFyY2hfcnVuX251bWJlcik7XG4gICAgICAgICAgc2V0RGF0YXNldE5hbWUgJiYgc2V0RGF0YXNldE5hbWUoZm9ybV9zZWFyY2hfZGF0YXNldF9uYW1lKTtcbiAgICAgICAgfX1cbiAgICAgID5cbiAgICAgICAgPEZvcm0uSXRlbSB7Li4udGFpbExheW91dH0+XG4gICAgICAgICAgPFF1ZXN0aW9uQnV0dG9uIC8+XG4gICAgICAgIDwvRm9ybS5JdGVtPlxuICAgICAgICA8U3R5bGVkRm9ybUl0ZW0gbmFtZT1cInJ1bl9udW1iZXJcIj5cbiAgICAgICAgICA8U3R5bGVkSW5wdXRcbiAgICAgICAgICAgIGlkPVwicnVuX251bWJlclwiXG4gICAgICAgICAgICBvbkNoYW5nZT17KGU6IENoYW5nZUV2ZW50PEhUTUxJbnB1dEVsZW1lbnQ+KSA9PlxuICAgICAgICAgICAgICBzZXRGb3JtUnVuTnVtYmVyKGUudGFyZ2V0LnZhbHVlKVxuICAgICAgICAgICAgfVxuICAgICAgICAgICAgcGxhY2Vob2xkZXI9XCJFbnRlciBydW4gbnVtYmVyXCJcbiAgICAgICAgICAgIHR5cGU9XCJ0ZXh0XCJcbiAgICAgICAgICAgIG5hbWU9XCJydW5fbnVtYmVyXCJcbiAgICAgICAgICAgIHZhbHVlPXtkZWZhdWx0UnVuTnVtYmVyfVxuICAgICAgICAgIC8+XG4gICAgICAgIDwvU3R5bGVkRm9ybUl0ZW0+XG4gICAgICAgIDxTdHlsZWRGb3JtSXRlbSBuYW1lPVwiZGF0YXNldF9uYW1lXCI+XG4gICAgICAgICAgPFN0eWxlZElucHV0XG4gICAgICAgICAgICBpZD1cImRhdGFzZXRfbmFtZVwiXG4gICAgICAgICAgICBwbGFjZWhvbGRlcj1cIkVudGVyIGRhdGFzZXQgbmFtZVwiXG4gICAgICAgICAgICBvbkNoYW5nZT17KGU6IENoYW5nZUV2ZW50PEhUTUxJbnB1dEVsZW1lbnQ+KSA9PlxuICAgICAgICAgICAgICBzZXRGb3JtRGF0YXNldE5hbWUoZS50YXJnZXQudmFsdWUpXG4gICAgICAgICAgICB9XG4gICAgICAgICAgICB0eXBlPVwidGV4dFwiXG4gICAgICAgICAgICB2YWx1ZT17ZGVmYXVsdERhdGFzZXROYW1lfVxuICAgICAgICAgIC8+XG4gICAgICAgIDwvU3R5bGVkRm9ybUl0ZW0+XG4gICAgICAgIDxGb3JtLkl0ZW0gey4uLnRhaWxMYXlvdXR9PlxuICAgICAgICAgIDxTZWFyY2hCdXR0b25cbiAgICAgICAgICAgIG9uQ2xpY2s9eygpID0+XG4gICAgICAgICAgICAgIGhhbmRsZXIoZm9ybV9zZWFyY2hfcnVuX251bWJlciwgZm9ybV9zZWFyY2hfZGF0YXNldF9uYW1lKVxuICAgICAgICAgICAgfVxuICAgICAgICAgIC8+XG4gICAgICAgIDwvRm9ybS5JdGVtPlxuICAgICAgPC9DdXN0b21Gb3JtPlxuICAgIDwvZGl2PlxuICApO1xufTtcblxuZXhwb3J0IGRlZmF1bHQgTmF2O1xuIl0sInNvdXJjZVJvb3QiOiIifQ==